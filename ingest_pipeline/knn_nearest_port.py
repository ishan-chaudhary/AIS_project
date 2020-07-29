#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 07:49:08 2020

@author: patrickmaus
"""

import gsta
import gsta_config

import pandas as pd
import numpy as np
from datetime import datetime

from psycopg2.extras import execute_values

from sklearn.neighbors import BallTree
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

import importlib
importlib.reload(gsta)

conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.colone_cargo_params)
conn.close()

#%% get the sits as a df from the database
sites = gsta.get_sites(loc_engine)

#%% Create "nearest_site" table in the database.
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
c = conn.cursor()
c.execute("""DROP TABLE IF EXISTS nearest_site""")
conn.commit()
c.execute("""CREATE TABLE IF NOT EXISTS nearest_site
(   id serial,
    nearest_port_id int,
    nearest_port_dist_km float
);""")
conn.commit()
c.close()
conn.close()
#%% production version.  uses balltree with dual tree and psycopg's executemany
# using pandas.to_sql() took 19 hours.  This implementation took 6.5 hours for 207,178,914 rows
start = datetime.now()
# establish the connection
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)

# build the BallTree using the ports as the candidates
candidates = np.radians(sites.loc[:, ['lat', 'lon']].values)
tree = BallTree(candidates, leaf_size=40, metric='haversine')

read_sql = """SELECT id, lat, lon
            FROM uid_positions;"""
chunksize = 500000
count = 0
generator = pd.read_sql(sql=read_sql, con=loc_engine, chunksize=chunksize)

for df in generator:
    iteration_start = datetime.now()
    # Now we are going to use sklearn's BallTree to find the nearest neighbor of
    # each position for the nearest port.  The resulting port_id and dist will be
    # pushed back to the db with the id, uid, and time to be used in the network
    # building phase of analysis.  This takes up more memory, but means we have
    # fewer joins.  Add an index on uid though before running network building.
    # transform to radians
    points_of_int = np.radians(df.loc[:, ['lat', 'lon']].values)
    # query the tree
    dist, ind = tree.query(points_of_int, k=1, dualtree=True)
    # make the data list to pass to the sql query
    data = np.column_stack((np.round(((dist.reshape(1, -1)[0]) * 6371.0088), decimals=3),
                            sites.iloc[ind.reshape(1, -1)[0], :].port_id.values.astype('int'),
                            df['id'].values))
    # define the sql statement
    sql_insert = "INSERT INTO nearest_site (nearest_port_dist_km, nearest_port_id, id) " \
                 "VALUES(%s, %s, %s);"

    # write to db
    c = conn.cursor()
    c.executemany(sql_insert, (data.tolist()))
    conn.commit()
    c.close()

    count += len(df)
    print('Iteration complete in:', datetime.now() - iteration_start)
    print(f"{count:,d} total rows written.")

print('Total Lapse:', datetime.now()-start)
conn.close()

#%% build index
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
c = conn.cursor()
c.execute("""CREATE INDEX if not exists nearest_site_uid_idx 
            on nearest_site (id);""")
conn.commit()
conn.close()
