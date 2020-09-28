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

#%%
start_time = '2017-01-01 00:00:00'
end_time = '2017-02-01 00:00:00'

# %% Create needed accessory tables and ensure they are clean.  also get uid list
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
c = conn.cursor()

#Create "clustering_results" table in the database.
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
    return uids_len[0] # remember its a tuple from the db.  [0] gets the int


# %%
def sklearn_dbscan(uid, eps, min_samp, print_verbose=False):
    iteration_start = datetime.datetime.now()
    temp_table_name = f'temp_{str(uid[0])}'

    engine_pg = gsta.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)
    conn_pg = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    c_pg = conn_pg.cursor()

    # next get the data for the uid
    read_sql = f"""SELECT id, lat, lon
                FROM uid_positions
                WHERE uid = '{uid[0]}'
                AND time between '{start_time}' and '{end_time}'
                ORDER by time"""
    df = pd.read_sql_query(read_sql, con=engine_pg)

    # format data for dbscan
    X = (np.radians(df.loc[:, ['lon', 'lat']].values))
    x_id = df.loc[:, 'id'].values

    try:
        if method == 'dbscan':
            # execute sklearn's DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samp, algorithm='ball_tree',
                            metric='haversine', n_jobs=1)
            dbscan.fit(X)
            results_dict = {'id': x_id, 'clust_id': dbscan.labels_}

        if method == 'optics':
            # execute sklearn's OPTICS
            # 5km in radians is max eps
            optics = OPTICS(max_eps=eps, min_samples=min_samp,  metric='euclidean', cluster_method='xi',
                            algorithm='kd_tree', n_jobs=1)
            optics.fit(X)
            results_dict = {'id': x_id, 'clust_id': optics.labels_}

        else:
            print("Error.  Method must be 'dbscan' or 'optics'.")
            return

        # gather the output as a dataframe
        df_results = pd.DataFrame(results_dict)
        # drop all -1 clust_id, which are all points not in clusters
        df_results = df_results[df_results['clust_id'] != -1]
        # write results to database in a temp table with the uid in the name
        sql_drop_table = f"""DROP TABLE IF EXISTS {temp_table_name};"""
        c_pg.execute(sql_drop_table)
        conn_pg.commit()
        sql_create_table = f"""CREATE TEMPORARY TABLE {temp_table_name}
                           (id int, 
                           clust_id int);"""
        c_pg.execute(sql_create_table)
        conn_pg.commit()
        df_results.to_sql(name=temp_table_name, con=engine_pg,
                          if_exists='append', method='multi', index=False)
        # take the clust_ids from the temp table and insert them into the temp table
        sql_update = f"UPDATE clustering_results AS c " \
                     f"SET {params_name} = clust_id " \
                     f"FROM {temp_table_name} AS t WHERE t.id = c.id"
        c_pg.execute(sql_update)
        conn_pg.commit()
        c_pg.close()

    except Exception as e:
        print(f'UID {uid[0]} error in clustering or writing clustering results to the database.')
        print(e)

    # delete the temp table
    c_pg = conn_pg.cursor()
    c_pg.execute(f'DROP TABLE IF EXISTS {temp_table_name};')
    conn_pg.commit()
    # add the uid to the tracker and get current uid count from tracker
    uids_completed = add_to_uid_tracker(uid, conn_pg)
    c_pg.close()

    # delete the connections
    engine_pg.dispose()
    conn_pg.close()

    if print_verbose == True:
        print(f'UID {uid[0]} complete in ', datetime.datetime.now() - iteration_start)
        percentage = (uids_completed / len(uid_list)) * 100
        print(f'Approximately {round(percentage, 3)} complete this run.')


def postgres_dbscan(uid, eps, min_samp):
    """
    A function to conduct dbscan on the server for a global eps and min_samples value.
    Optimized for multiprocessing.
    :param min_samp:
    :param eps:
    :param uid:
    :return:
    """
    #iteration_start = datetime.datetime.now()
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
epsilons = [5]
min_samples = [25]
#min_samples = [25, 50, 100, 200, 300, 400, 500]
method = 'optics'

for eps_km in epsilons:
    for min_samp in min_samples:
        iteration_start = datetime.datetime.now()
        eps = eps_km / 6371.0088 #kms per radian
        params_name = f"{method}_{str(eps_km).replace('.','_')}_{min_samp}"
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
                p.starmap(sklearn_dbscan, zip(uid_list, repeat(eps), repeat(min_samp)))
            except Exception as e:
                print (e)


        print(f'Method {params_name} complete in ', datetime.datetime.now() - iteration_start)

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
