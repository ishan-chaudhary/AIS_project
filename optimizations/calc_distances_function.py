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

from sklearn.neighbors import BallTree

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_full_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_full_params)
ports_wpi = gsta.get_ports_wpi(loc_engine)

# %% Create "nearest_port" table in the database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS nearest_port
(   id int,
    uid text,
    time timestamp,
    nearest_port_id int,
    nearest_port_dist_km float
);""")
conn.commit()
c.close()

#%%
start = datetime.now()

# build the BallTree using the ports as the candidates
candidates = np.radians(ports_wpi.loc[:, ['lat', 'lon']].values)
tree = BallTree(candidates, leaf_size=40, metric='haversine')

# create the pandas generator using chuncksize
from_schema_name = 'public'
source_table = 'cargo_ship_position'
new_table_name = 'nearest_port'
read_sql = """SELECT id, uid, time, lat, lon
            FROM {0}.{1};
            """.format(from_schema_name, source_table)
chunksize = 100000
count = 0
generator = pd.read_sql(sql=read_sql, con=loc_engine, chunksize=chunksize)

for df in generator:
    iteration_start = datetime.now()
    # Now we are going to use sklearn's BallTree to find the nearest neighbor of
    # each position for the nearest port.  The resulting port_id and dist will be
    # pushed back to the db with the id, uid, and time to be used in the network
    # building phase of analysis.  This takes up more memory, but means we have
    # fewer joins.  Add an index on uid though before running network building.
    points_of_int = np.radians(df.loc[:, ['lat', 'lon']].values)
    nearest_list = []
    for i in range(len((points_of_int))):
        dist, ind = tree.query(points_of_int[i,:].reshape(1, -1), k=1)
        nearest_dict ={'id':df['id'].iloc[i],
                       'uid':df['uid'].iloc[i],
                       'time': df['time'].iloc[i],
                       'nearest_port_id':ports_wpi.iloc[ind[0][0]].loc['port_id'],
                       'nearest_port_dist_km':dist[0][0]*6371.0088}
        nearest_list.append(nearest_dict)
    df_nearest = pd.DataFrame(nearest_list)

    # write df to databse
    df_nearest.to_sql(name=new_table_name, con=loc_engine,
                      if_exists='append', method='multi', index=False)
    count += len(df)
    print('Iteration complete in:', datetime.now() - iteration_start)
    print(f"{count:,d} total rows written.")

print('Total Lapse:', datetime.now()-start)