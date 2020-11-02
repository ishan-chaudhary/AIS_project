import datetime
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
import os

import gsta_config
import gnact
from gnact import utils
from gnact import clust

import warnings

warnings.filterwarnings('ignore')

# %% set parameters
# id number of cores and set workers.  use n-1 workers to keep from crashing machine.
cores = os.cpu_count()
workers = cores - 1
print(f'This machine has {cores} cores.  Will use {workers} for multiprocessing.')

# set tables for processing
source_table = 'uid_positions_jan'
clustering_results_table = 'clustering_results'
clustering_times_table = 'clustering_times'

# to control date range from target table
start_time = '2017-01-01 00:00:00'
end_time = '2017-02-01 00:00:00'

# used in dbscan and stdbscan as eps, optics and hdbscan as max eps
epsilons_km = [3]
# used in all as minimum size of cluster
min_samples = [0]
# if not stdbscan, leave as sing integer in list.  values is minutes
epsilons_time = [240, 360, 480, 600, 720]

method = 'dynamic_segmentation'

# %% Create needed accessory tables and ensure they are clean.  also get uid list
conn = gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
c = conn.cursor()
# Create "clustering_results" table in the database.
c.execute(f"""CREATE TABLE IF NOT EXISTS {clustering_results_table}
        AS (SELECT id from {source_table}
        where time between '{start_time}' and '{end_time}'
        );""")
conn.commit()
print('clustering_results table exists.')

# make sure the index is created
#c.execute(f"""CREATE INDEX if not exists clustering_results_id_idx on {clustering_results_table} (id);""")
#conn.commit()
#print('Index on id in clustering_results exists.')

# get the uid list from the uid_trips table
c.execute(f"""SELECT DISTINCT(uid) FROM {source_table}
        where time between '{start_time}' and '{end_time}';""")
uid_list = c.fetchall()
print(f'{str(len(uid_list))} total uids returned.')

# Create "clustering_times" table in the database.  Unique name constraint needed for upsert so new runs overwrite olds.
c.execute(f"""CREATE TABLE IF NOT EXISTS {clustering_times_table}
(run timestamp,
name text UNIQUE,
time_diff interval);""")
conn.commit()
print('clustering_times table exists.')

c.close()
conn.close()


# %%
def pooled_clustering(uid, eps_km, min_samp, eps_time=None, method=None, print_verbose=True):
    start = datetime.datetime.now()
    if eps_time is None:
        table_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}"
    else:
        table_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}_{eps_time}"
    # create db connections within the loop
    engine_pg = gnact.utils.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)
    # get the clustering results
    try:
        # get the positions for the uid, and cluster them
        df_posits = gnact.clust.get_uid_posits(uid, engine_pg, end_time=end_time)
        df_results = gnact.clust.get_clusters(df_posits, eps_km=eps_time, min_samp=min_samp, eps_time=eps_time,
                                              method=method)
        # write the results to the db
        if type(pd.DataFrame()) == type(df_results) and len(df_results) > 0:
            df_results[['id', 'clust_id']].to_sql(name=table_name, con=engine_pg, if_exists='append',
                                                  method='multi', index=False)
    except Exception as e:
        print(f'UID {uid[0]} error in clustering or writing results to db.')
        print(e)
    if print_verbose:
        print(f'UID {uid[0]} complete in {datetime.datetime.now() - start} with {len(df_results)} rows added.')
        # with gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False) as conn_pg:
        #     uids_completed = gnact.utils.add_to_uid_tracker(uid, conn_pg)
        # conn_pg.close()
        # percentage = (uids_completed / len(uid_list)) * 100
        # print(f'Approximately {round(percentage, 3)} complete for {eps_km} km eps, {eps_time} for time, '
        #       f'and {min_samp} min sample run.')
    # close the connections
    engine_pg.dispose()


# %%
first_tick = datetime.datetime.now()
print()
print('Starting Processing at: ', first_tick.time())
for eps_km in epsilons_km:
    for min_samp in min_samples:
        for eps_time in epsilons_time:
            iteration_start = datetime.datetime.now()
            # Set the params_name based on the method
            if method in ['dbscan']:
                params_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}"
            elif method in ['optics', 'hdbscan']:
                params_name = f"{method}_{str(eps_km).replace('.', '_')}"
            elif method in ['stdbscan', 'dynamic_segmentation']:
                params_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}_{eps_time}"
            else:
                print("Method must be one of 'dbscan', 'optics', 'hdbscan', stdbscan', or dynamic segmentation.")
                break
            print(f'Starting processing for {params_name}...')

            try:
                # create db connections for outside the loop
                with gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False) as conn:
                    with conn.cursor() as c:
                        # create a table for this run to store all of the ids and the clust_id
                        sql_drop_table = f"""DROP TABLE IF EXISTS {params_name};"""
                        c.execute(sql_drop_table)
                        conn.commit()
                        sql_create_table = f"""CREATE TABLE {params_name}
                                           (id int, 
                                           clust_id int);"""
                        c.execute(sql_create_table)
                        conn.commit()
                        # make the uid tracker
                        gnact.utils.make_uid_tracker(conn)
                        print('Connections and tables created.')
            except Exception as e:
                print('Error creating connections or tables.')
                print(e)
                break
            finally:
                if conn is not None:
                    conn.close()

            # execute the function with pooled workers.  This will populate the empty table
            with Pool(workers) as p:
                try:
                    p.starmap(pooled_clustering, zip(uid_list, repeat(eps_km), repeat(min_samp),
                                                     repeat(eps_time), repeat(method)))
                except Exception as e:
                    print('Error in pooling:', e)
            print(f'Finished pooled clustering at {datetime.datetime.now()}')

            # make sure the method name column exists and is clear
            try:
                with gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False) as conn:
                    with conn.cursor() as c:
                        c.execute(f"""ALTER TABLE {clustering_results_table} DROP COLUMN IF EXISTS
                                        {params_name};""")
                        conn.commit()
                        c.execute(f"""ALTER TABLE {clustering_results_table} ADD COLUMN IF NOT EXISTS
                                    {params_name} int;""")
                        conn.commit()
                        print(f'Clean column for {params_name} exists at {datetime.datetime.now()}.')
            except Exception as e:
                print('Error in cleaning column for run.')
                print(e)
            finally:
                if conn is not None:
                    conn.close()
            # # add foreign keys to speed up the join
            # try:
            #     print('Adding foreign keys ...')
            #     c.execute(f"""ALTER TABLE {params_name} ADD CONSTRAINT id_to_id
            #     FOREIGN KEY (id) REFERENCES  (id)""")
            #     conn.commit()
            #     print(f'Foreign keys added at {datetime.datetime.now()}.')
            # except Exception as e:
            #     print("Error building foreign key.")
            #     print(e)

            # take the clust_ids from the temp table and insert them into the clustering_results table

            try:
                print('Updating results table...')
                sql_update = f"UPDATE {clustering_results_table} AS c " \
                             f"SET {params_name} = clust_id " \
                             f"FROM {params_name} AS t WHERE t.id = c.id"
                with gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False) as conn:
                    with conn.cursor() as c:
                        c.execute(sql_update)
                        conn.commit()
                        print(f'Results table updated at {datetime.datetime.now()}.')
            except Exception as e:
                print('Unable to update clustering results table.')
                print(e)
            finally:
                if conn is not None:
                    conn.close()

            # get the total time of the iteration
            iteration_stop = datetime.datetime.now()
            iteration_lapse = iteration_stop - iteration_start
            print(f'Method {params_name} complete in {iteration_lapse}.')

            # log the time for the iteration
            try:
                sql_upsert = f"""INSERT INTO {clustering_times_table} (run, name, time_diff) 
                                 VALUES ('{first_tick.strftime('%Y-%m-%d %H:%M:%S')}', '{params_name}', '{iteration_lapse}')
                                 ON CONFLICT (name) DO UPDATE 
                                     SET run = excluded.run, 
                                     time_diff = excluded.time_diff;"""
                with gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False) as conn:
                    with conn.cursor() as c:
                        c.execute(sql_upsert)
                        conn.commit()
                        print(f'Run logged at {datetime.datetime.now()}.')
            except Exception as e:
                print('Error logging times for run.')
                print(e)
            finally:
                if conn is not None:
                    conn.close()

            # delete the temp table
            try:
                with gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False) as conn:
                    with conn.cursor() as c:
                        c.execute(sql_drop_table)
                        conn.commit()
            except Exception as e:
                print('Error deleting temp table.')
                print(e)
            finally:
                if conn is not None:
                    conn.close()
            print()

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)
