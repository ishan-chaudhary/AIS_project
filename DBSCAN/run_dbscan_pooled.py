
import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from multiprocessing import Pool

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

from importlib import reload
reload(gsta)


#%% Create "clustering_results" table in the database.
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS clustering_results 
        AS (SELECT id from uid_positions);""")
conn.commit()
c.close()
print('clustering_results table exists.')

c = conn.cursor()
c.execute("""CREATE INDEX if not exists clustering_results_id_idx 
            on clustering_results (id);""")
conn.commit()
print('Index on id in clustering_results exists.')

# get the uid list from the uid_trips table
c = conn.cursor()
c.execute(f"""SELECT DISTINCT(uid) FROM uid_trips;""")
uid_list = c.fetchall()
print(f'{str(len(uid_list))} total uids returned.')
c.close()

eps_km = 2
kms_per_radian = 6371.0088
eps = eps_km / kms_per_radian
min_samples = 250
method = 'dbscan'

c = conn.cursor()
c.execute(f"""ALTER TABLE clustering_results ADD COLUMN IF NOT EXISTS
            {method}_{eps_km}_{min_samples} int;""")
conn.commit()
print(f'column {method}_{eps_km}_{min_samples} exists.')
c.execute(f"""UPDATE clustering_results 
            SET {method}_{eps_km}_{min_samples} = NULL
            WHERE {method}_{eps_km}_{min_samples} IS NOT NULL;""")
conn.commit()
print(f'column {method}_{eps_km}_{min_samples} is cleared.')
c.close()
conn.close()


#%%
def sklearn_dbscan(uid):
    print('Working on uid:', uid[0])
    iteration_start = datetime.datetime.now()
    engine = gsta.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)
    # next get the data for the uid
    read_sql = f"""SELECT id, lat, lon
                FROM uid_positions
                WHERE uid = '{uid[0]}'
                ORDER by time"""
    df = pd.read_sql_query(read_sql, con=engine)

    # format data for dbscan
    X = (np.radians(df.loc[:, ['lon', 'lat']].values))
    x_id = df.loc[:, 'id'].values

    # execute sklearn's DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree',
                    metric='haversine', n_jobs=1)
    dbscan.fit(X)

    # gather the output as a dataframe
    results_dict = {'id': x_id, 'clust_id': dbscan.labels_}
    df_results = pd.DataFrame(results_dict)
    # drop all -1 clust_id, which are all points not in clusters
    df_results = df_results[df_results['clust_id'] != -1]
    # write results to database in a temp table with the uid in the name
    df_results.to_sql(name=f'temp_{str(uid[0])}', con=engine,
                      if_exists='replace', method='multi', index=False)
    engine.dispose()

    # take the clust_ids from the temp table and insert them into the
    conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    sql_update = f"UPDATE clustering_results AS c " \
                 f"SET {method}_{eps_km}_{min_samples} = clust_id " \
                 f"FROM temp_{str(uid[0])} AS t WHERE t.id = c.id"
    c = conn.cursor()
    c.execute(sql_update)
    conn.commit()

    c.execute(f'DROP TABLE temp_{str(uid[0])};')
    conn.commit()
    c.close()
    conn.close()

    print(f'UID {uid[0]} complete in:', datetime.datetime.now() - iteration_start)


#%%
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

# execute the function with pooled workers
if __name__ == '__main__':
    with Pool(40) as p:
        p.map(sklearn_dbscan, uid_list)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)
conn.close()
#%%
