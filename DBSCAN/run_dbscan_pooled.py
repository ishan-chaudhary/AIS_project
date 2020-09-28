import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from multiprocessing import Pool
from itertools import repeat

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

from importlib import reload
reload(gsta)

import warnings
warnings.filterwarnings('ignore')

# %%
start_time = '2017-01-01 00:00:00'
end_time = '2018-01-01 00:00:00'

# %% Create needed accessory tables and ensure they are clean.  also get uid list
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
c = conn.cursor()

# Create "clustering_results" table in the database.
c.execute(f"""CREATE TABLE IF NOT EXISTS clustering_results 
        AS (SELECT id from uid_positions
        where time between '{start_time}' and '{end_time}'
        );""")
conn.commit()
print('clustering_results table exists.')

# make sure the index is created
c.execute("""CREATE INDEX if not exists clustering_results_id_idx 
            on clustering_results (id);""")
conn.commit()
print('Index on id in clustering_results exists.')

# get the uid list from the uid_trips table
c.execute(f"""SELECT DISTINCT(uid) FROM uid_positions
        where time between '{start_time}' and '{end_time}';""")
uid_list = c.fetchall()
print(f'{str(len(uid_list))} total uids returned.')

c.close()
conn.close()


# %%
def make_uid_tracker(conn_pg):
    """
    This function makes a tracking table for the UIDs already processed in a script
    :param conn_pg:
    :return: a new table is created (or dropped and recreated) at the end of the conn
    """
    c_pg = conn_pg.cursor()
    c_pg.execute("""DROP TABLE IF EXISTS uid_tracker""")
    conn_pg.commit()
    c_pg.execute("""CREATE TABLE IF NOT EXISTS uid_tracker
    (uid text);""")
    conn_pg.commit()
    c_pg.close()
    print('Clean UID tracker table created.')


def add_to_uid_tracker(uid, conn_pg):
    """
    This function adds a provided uid to the tracking database and returns the len of
    uids already in the table.
    :param uid: tuple, from the returned uid list from the db
    :param conn_pg:
    :return: an int the len of distinct uids in the uid_tracker table
    """
    c_pg = conn_pg.cursor()
    # track completed uids by writing to a new table
    insert_uid_sql = """INSERT INTO uid_tracker (uid) values (%s)"""
    c_pg.execute(insert_uid_sql, uid)
    conn_pg.commit()
    # get total number of uids completed
    c_pg.execute("""SELECT count(distinct(uid)) from uid_tracker""")
    uids_len = (c_pg.fetchone())
    c_pg.close()
    conn_pg.close()
    return uids_len[0]  # remember its a tuple from the db.  [0] gets the int


def get_uid_posits(uid, engine_pg, start_time='2017-01-01 00:00:00', end_time='2018-01-01 00:00:00'):
    read_sql = f"""SELECT id, lat, lon
                FROM uid_positions
                WHERE uid = '{uid[0]}'
                AND time between '{start_time}' and '{end_time}'
                ORDER by time"""
    df_posits = pd.read_sql_query(read_sql, con=engine_pg)
    return df_posits


def get_clusters(df, eps_km, min_samp, method):
    # format data for dbscan
    X = (np.radians(df.loc[:, ['lon', 'lat']].values))
    x_id = df.loc[:, 'id'].values
    try:
        if method == 'dbscan':
            # execute sklearn's DBSCAN
            dbscan = DBSCAN(eps=eps_km/6371, min_samples=min_samp, algorithm='ball_tree',
                            metric='haversine', n_jobs=1)
            dbscan.fit(X)
            results_dict = {'id': x_id, 'clust_id': dbscan.labels_}
        if method == 'optics':
            # execute sklearn's OPTICS
            # 5km in radians is max eps
            optics = OPTICS(max_eps=eps_km/6371, min_samples=min_samp, metric='euclidean', cluster_method='xi',
                            algorithm='kd_tree', n_jobs=1)
            optics.fit(X)
            results_dict = {'id': x_id, 'clust_id': optics.labels_}
        if method not in ['optics', 'dbscan']:
            print("Error.  Method must be 'dbscan' or 'optics'.")
            return None
    except Exception as e:
        print(f'UID {uid[0]} error in clustering.')
        print(e)
        return None
    # gather the output as a dataframe
    df_results = pd.DataFrame(results_dict)
    # drop all -1 clust_id, which are all points not in clusters
    df_results = df_results[df_results['clust_id'] != -1]
    return df_results


#%%
def pooled_clustering(uid, eps_km, min_samp, method, print_verbose=True):
    iteration_start = datetime.datetime.now()
    dest_column = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}"
    temp_table_name = f'temp_{str(uid[0])}'

    engine_pg = gsta.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)
    conn_pg = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    c_pg = conn_pg.cursor()

    df_posits = get_uid_posits(uid, engine_pg)
    df_results = get_clusters(df_posits, eps_km=eps_km, min_samp=min_samp, method=method)

    try:
        # write results to database in a temp table with the uid in the name
        sql_drop_table = f"""DROP TABLE IF EXISTS {temp_table_name};"""
        c_pg.execute(sql_drop_table)
        conn_pg.commit()
        sql_create_table = f"""CREATE TABLE {temp_table_name}
                           (id int, 
                           clust_id int);"""
        c_pg.execute(sql_create_table)
        conn_pg.commit()
        df_results.to_sql(name=temp_table_name, con=engine_pg,
                          if_exists='append', method='multi', index=False)
        # take the clust_ids from the temp table and insert them into the temp table
        sql_update = f"UPDATE clustering_results AS c " \
                     f"SET {dest_column} = clust_id " \
                     f"FROM {temp_table_name} AS t WHERE t.id = c.id"
        c_pg.execute(sql_update)
        conn_pg.commit()

    except Exception as e:
        print(f'UID {uid[0]} error in writing clustering results to the database.')
        print(e)

    # delete the temp table
    c_pg.execute(sql_drop_table)
    conn_pg.commit()
    c_pg.close()
    # close the connections
    engine_pg.dispose()
    conn_pg.close()

    if print_verbose == True:
        print(f'UID {uid[0]} complete in ', datetime.datetime.now() - iteration_start)
        uids_completed = add_to_uid_tracker(uid, conn_pg)
        percentage = (uids_completed / len(uid_list)) * 100
        print(f'Approximately {round(percentage, 3)} complete this run.')

def postgres_dbscan_reworked(uid, eps_km, min_samp):
    """
    A function to conduct dbscan on the server for a global eps and min_samples value.
    Optimized for multiprocessing.
    :param min_samp:
    :param eps:
    :param uid:
    :return:
    """
    # iteration_start = datetime.datetime.now()
    # execute dbscan script
    dbscan_postgres_sql = f"""
    UPDATE clustering_results as c 
    SET {params_name} = t.clust_id
    FROM (SELECT id , ST_ClusterDBSCAN(geom, eps := {eps}, minpoints := {min_samp})
          over () as clust_id
          FROM uid_positions
          WHERE uid = '{uid[0]}'
          AND time between '{start_time}' and '{end_time}') as t
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
# for optics, just put the max eps in for list of epsilons.
epsilons_km = [5]
min_samples = [25, 50, 100, 200, 300, 400, 500]
method = 'optics'


for eps_km in epsilons_km:
    for min_samp in min_samples:
        iteration_start = datetime.datetime.now()
        params_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}"
        print(f'Starting processing for {params_name}...')

        conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
        make_uid_tracker(conn)
        c = conn.cursor()
        # make sure the method name column exists and is clear
        c.execute(f"""ALTER TABLE clustering_results DROP COLUMN IF EXISTS
                    {params_name};""")
        conn.commit()
        c.execute(f"""ALTER TABLE clustering_results ADD COLUMN IF NOT EXISTS
                    {params_name} int;""")
        conn.commit()
        c.close()
        print(f'Clean column for {params_name} exists.')
        conn.close()

        # execute the function with pooled workers
        with Pool(20) as p:
            try:
                 p.starmap(pooled_clustering, zip(uid_list, repeat(eps_km), repeat(min_samp), repeat(method)))
            except Exception as e:
                print(e)

        print(f'Method {params_name} complete in ', datetime.datetime.now() - iteration_start)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

