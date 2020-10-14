import datetime
from multiprocessing import Pool
from itertools import repeat
import pandas as pd

import gsta_config
import gnact
from gnact import utils
from gnact import clust

from importlib import reload
import warnings

warnings.filterwarnings('ignore')

# noinspection PyTypeChecker
reload(gnact)

# %%
start_time = '2017-01-01 00:00:00'
end_time = '2017-02-01 00:00:00'

# %% Create needed accessory tables and ensure they are clean.  also get uid list
conn = gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
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

# Create "clustering_times" table in the database.
c.execute(f"""CREATE TABLE IF NOT EXISTS clustering_times
(run timestamp,
name text,
seconds int);""")
conn.commit()
print('clustering_times table exists.')

c.close()
conn.close()


# %%
def pooled_clustering(uid, eps_km, min_samp, eps_time, method, print_verbose=True):
    iteration_start = datetime.datetime.now()
    if eps_time == None:
        table_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}"
    else:
        table_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}_{eps_time}"
    # create db connections within the loop
    engine_pg = gnact.utils.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)
    conn_pg = gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    c_pg = conn_pg.cursor()
    # get the clustering results
    try:
        # get the positions for the uid, and cluster them
        df_posits = gnact.clust.get_uid_posits(uid, engine_pg, end_time=end_time)
        df_results = gnact.clust.get_clusters(df_posits, eps_km=eps_km, min_samp=min_samp, time_window=eps_time, method=method)
        # write the results to the db
        if type(df_results) == type(pd.DataFrame()):
            df_results[['id', 'clust_id']].to_sql(name=table_name, con=engine_pg,
                              if_exists='append', method='multi', index=False)
    except Exception as e:
        print(f'UID {uid[0]} error in clustering or writing results to db.')
        print(e)
    if print_verbose:
        print(f'UID {uid[0]} complete in ', datetime.datetime.now() - iteration_start)
        uids_completed = gnact.utils.add_to_uid_tracker(uid, conn_pg)
        percentage = (uids_completed / len(uid_list)) * 100
        print(f'Approximately {round(percentage, 3)} complete for {eps_km} km eps, {eps_time} for time, and {min_samp} min sample run.')
    # close the connections
    c_pg.close()
    conn_pg.close()
    engine_pg.dispose()


# %%
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())
# for optics, just put the max eps in for list of epsilons.
epsilons_km = [1, 3]
min_samples = [50, 100, 200, 300]
epsilons_time = [360, 600, 840]
method = 'stdbscan'

for eps_km in epsilons_km:
    for min_samp in min_samples:
        for eps_time in epsilons_time:
            iteration_start = datetime.datetime.now()
            if eps_time == None:
                params_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}"
            else:
                params_name = f"{method}_{str(eps_km).replace('.', '_')}_{min_samp}_{eps_time}"
            print(f'Starting processing for {params_name}...')
            # create db connections for outside the loop
            conn = gnact.utils.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
            c = conn.cursor()
            engine = gnact.utils.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)
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

            # execute the function with pooled workers.  This will populate the empty table
            with Pool(39) as p:
                try:
                    p.starmap(pooled_clustering, zip(uid_list, repeat(eps_km), repeat(min_samp), repeat(eps_time), repeat(method)))
                except Exception as e:
                    print('Error in pooling:', e)
            print(f'Finished pooled clustering at {datetime.datetime.now()}')
            # make sure the method name column exists and is clear
            c.execute(f"""ALTER TABLE clustering_results DROP COLUMN IF EXISTS
                        {params_name};""")
            conn.commit()
            c.execute(f"""ALTER TABLE clustering_results ADD COLUMN IF NOT EXISTS
                        {params_name} int;""")
            conn.commit()
            print(f'Clean column for {params_name} exists at {datetime.datetime.now()}.')

            # add foreign keys to speed up the join
            try:
                print('Adding foreign keys ...')
                c.execute(f"""ALTER TABLE {params_name} ADD CONSTRAINT id_to_id 
                FOREIGN KEY (id) REFERENCES clustering_results (id)""")
                conn.commit()
                print(f'Foreign keys added at {datetime.datetime.now()}.')
            except Exception as e:
                print("Error building foreign key.")
                print(e)
            print('Updating clustering_results table...')
            # take the clust_ids from the temp table and insert them into the temp table
            sql_update = f"UPDATE clustering_results AS c " \
                         f"SET {params_name} = clust_id " \
                         f"FROM {params_name} AS t WHERE t.id = c.id"
            c.execute(sql_update)
            conn.commit()
            print(f'Table updated at {datetime.datetime.now()}.')
            # delete the temp table
            c.execute(sql_drop_table)
            conn.commit()

            # close db connections
            c.close()
            conn.close()
            engine.dispose()

            print(f'Method {params_name} complete in ', datetime.datetime.now() - iteration_start)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)
