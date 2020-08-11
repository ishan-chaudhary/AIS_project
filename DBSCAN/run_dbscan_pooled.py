import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from multiprocessing import Pool

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

from importlib import reload

reload(gsta)

# %% Create "clustering_results" table in the database.
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
# %%
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
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
params_name = f'{method}_{eps_km}_{min_samples}'

c = conn.cursor()
c.execute(f"""ALTER TABLE clustering_results DROP COLUMN IF EXISTS
            {params_name};""")
conn.commit()
c.execute(f"""ALTER TABLE clustering_results ADD COLUMN IF NOT EXISTS
            {params_name} int;""")
conn.commit()
print(f'Clean column for {params_name} exists.')
c.close()


# %%
def make_uid_tracker(conn_pg):
    c_pg = conn_pg.cursor()
    c_pg.execute("""DROP TABLE IF EXISTS uid_tracker""")
    conn_pg.commit()
    c_pg.execute("""CREATE TABLE IF NOT EXISTS uid_tracker
    (uid text);""")
    conn_pg.commit()
    c_pg.close()
    print('Clean UID tracker table created.')


def add_to_uid_tracker(uid, conn_pg):
    c_pg = conn_pg.cursor()
    # track completed uids by writing to a new table
    insert_uid_sql = """INSERT INTO uid_tracker (uid) values (%s)"""
    c_pg.execute(insert_uid_sql, uid)
    conn_pg.commit()

    # get total number of uids completed
    c_pg.execute("""SELECT count(uid) from uid_tracker""")
    uids_completed = (c_pg.fetchone())
    c_pg.close()
    conn_pg.close()
    return uids_completed[0]


# %%
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

    if method == 'dbscan':
        # execute sklearn's DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree',
                        metric='haversine', n_jobs=1)
        dbscan.fit(X)
        results_dict = {'id': x_id, 'clust_id': dbscan.labels_}

    # gather the output as a dataframe
    df_results = pd.DataFrame(results_dict)
    # drop all -1 clust_id, which are all points not in clusters
    df_results = df_results[df_results['clust_id'] != -1]
    # write results to database in a temp table with the uid in the name
    df_results.to_sql(name=f'temp_{str(uid[0])}', con=engine,
                      if_exists='replace', method='multi', index=False)
    engine.dispose()

    # take the clust_ids from the temp table and insert them into the
    conn_pg = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    sql_update = f"UPDATE clustering_results AS c " \
                 f"SET {params_name} = clust_id " \
                 f"FROM temp_{str(uid[0])} AS t WHERE t.id = c.id"
    c_pg = conn_pg.cursor()
    c_pg.execute(sql_update)
    conn_pg.commit()

    c_pg.execute(f'DROP TABLE temp_{str(uid[0])};')
    conn_pg.commit()
    c_pg.close()
    # add the uid to the tracker and get current uid count from tracker
    uids_completed = add_to_uid_tracker(uid, conn_pg)
    conn_pg.close()

    print(f'UID {uid[0]} complete in ', datetime.datetime.now() - iteration_start)
    percentage = (uids_completed / len(uid_list)) * 100
    print(f'Approximately {round(percentage, 3)} complete.')


def postgres_dbscan(uid):
    print('Working on uid:', uid[0])
    iteration_start = datetime.datetime.now()
    # execute dbscan script
    dbscan_postgres_sql = f"""
    UPDATE clustering_results as c 
    SET {params_name} = t.clust_id
    FROM (SELECT id , ST_ClusterDBSCAN(geom, eps := {eps}, minpoints := {min_samples})
          over () as clust_id
          FROM uid_positions
          WHERE uid = '{uid[0]}') as t
          WHERE t.id = c.id
          AND t.clust_id IS NOT NULL;"""
    conn_pg = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    c_pg = conn_pg.cursor()
    c_pg.execute(dbscan_postgres_sql)
    conn_pg.commit()
    c_pg.close()
    # add the uid to the tracker and get current uid count from tracker
    uids_completed = add_to_uid_tracker(uid, conn_pg)
    conn_pg.close()

    print(f'UID {uid[0]} complete in ', datetime.datetime.now() - iteration_start)
    percentage = (uids_completed / len(uid_list)) * 100
    print(f'Approximately {round(percentage, 3)} complete.')


# %%
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
make_uid_tracker(conn)
conn.close()

# execute the function with pooled workers
if __name__ == '__main__':
    with Pool(38) as p:
        p.imap(postgres_dbscan, uid_list)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

#
# #%%
# conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
# uid = ('477478700',) # 40,000 rows is super fast as sklearn
# first_tick = datetime.datetime.now()
# print('Starting Processing at: ', first_tick.time())
# print('Working on uid:', uid[0])
# iteration_start = datetime.datetime.now()
# engine = gsta.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)
# # next get the data for the uid
# read_sql = f"""SELECT id, lat, lon
#              FROM uid_positions
#              WHERE uid = '{uid[0]}'
#              ORDER by time"""
# df = pd.read_sql_query(read_sql, con=engine)
#
# last_tock = datetime.datetime.now()
# lapse = last_tock - first_tick
# print('Processing Done.  Total time elapsed: ', lapse)
# #%%
# first_tick = datetime.datetime.now()
# print('Starting Processing at: ', first_tick.time())
#
# # format data for dbscan
# X = (np.radians(df.loc[:, ['lon', 'lat']].values))
# x_id = df.loc[:, 'id'].values
# #%%
# optics = OPTICS(min_samples=250, algorithm='ball_tree', metric='euclidean')
# optics.fit(X)
#
# #%%
#
# #%%
# # execute sklearn's DBSCAN
# dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree',
#                 metric='haversine', n_jobs=1)
# dbscan.fit(X)
# # gather the output as a dataframe
# results_dict = {'id': x_id, 'clust_id': dbscan.labels_}
#
# #%%
# df_results = pd.DataFrame(results_dict)
# # drop all -1 clust_id, which are all points not in clusters
# df_results = df_results[df_results['clust_id'] != 1]
#
# last_tock = datetime.datetime.now()
# lapse = last_tock - first_tick
# print('Processing Done.  Total time elapsed: ', lapse)
# #%%
# first_tick = datetime.datetime.now()
# print('Starting Processing at: ', first_tick.time())
#
# # write results to database in a temp table with the uid in the name
# df_results.to_sql(name=f'temp_{str(uid[0])}', con=engine,
#                   if_exists='replace', method='multi', index=False)
# engine.dispose()
#
# # take the clust_ids from the temp table and insert them into the
# conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
# sql_update = f"UPDATE clustering_results AS c " \
#              f"SET {method}_{eps_km}_{min_samples} = clust_id " \
#              f"FROM temp_{str(uid[0])} AS t WHERE t.id = c.id"
# c = conn.cursor()
# c.execute(sql_update)
# conn.commit()
#
# c.execute(f'DROP TABLE temp_{str(uid[0])};')
# conn.commit()
# c.close()
# conn.close()
#
# last_tock = datetime.datetime.now()
# lapse = last_tock - first_tick
# print('Processing Done.  Total time elapsed: ', lapse)
#
# print(f'UID {uid[0]} complete in:', datetime.datetime.now() - iteration_start)
