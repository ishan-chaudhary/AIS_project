#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:03:35 2020

@author: patrickmaus
"""

#time tracking
import datetime

# db admin
import psycopg2
from sqlalchemy import create_engine

import random

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

aws_conn = gsta.connect_psycopg2(gsta_config.aws_ais_cluster_params)
loc_conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
aws_conn.close()    



#%%
def postgres_dbscan(source_table, new_table_name, eps, min_samples, 
                    mmsi_list, conn, lat='lat', lon='lon'):
    
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS {}
    (
        id integer,
        mmsi text,
        lat numeric,
        lon numeric,
        clust_id int
    );""".format(new_table_name))
    conn.commit()

    print("""Starting processing on DBSCAN with eps={} and
          min_samples={} """.format(str(eps), str(min_samples)))
    
    for mmsi in mmsi_list:          
        dbscan_sql = """INSERT INTO {0} (id, mmsi, lat, lon, clust_id)
        SELECT id, mmsi, {1}, {2},
        ST_ClusterDBSCAN(geom, eps := {3}, minpoints := {4})
        over () as clust_id
        FROM {5}
        WHERE mmsi = '{6}';""".format(new_table_name, lat, lon, str(eps), str(min_samples), source_table, mmsi[0])
        
        # execute dbscan script
        c = conn.cursor()
        c.execute(dbscan_sql)
        conn.commit()
        c.close()
        
        print('MMSI {} complete.'.format(mmsi[0]))
    
    print('DBSCAN complete, {} created'.format(new_table_name))


def make_tables_geom(table, conn):
    # add a geom column to the new table and populate it from the lat and lon columns
    c = conn.cursor()
    c.execute("""ALTER TABLE {} ADD COLUMN
                geom geometry(Point, 4326);""".format(table))
    conn.commit()
    c.execute("""UPDATE {} SET
                geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);""".format(table))
    conn.commit()
    c.close()


#%%
drop = False
source_table = 'cargo_ship_position'

c = loc_conn.cursor()
c.execute("""SELECT DISTINCT(mmsi) FROM {};""".format(source_table))
mmsi_list = c.fetchall()
c.close()

print('Function run at:', datetime.datetime.now())
epsilons = [500, 1000, 2000, 5000]
samples = [10, 25, 50, 100, 250, 500]

for eps in epsilons:
    for min_samples in samples:        
        tick = datetime.datetime.now()
        # make the new table name
        new_table_name = ('dbscan_results_by_mmsi_' + str(eps).replace('.','_') +
                          '_' + str(min_samples))
        if drop == True:
            # drop table if exists if needed.
            c = loc_conn.cursor()
            c.execute("""DROP TABLE IF EXISTS {}""".format(new_table_name))
            loc_conn.commit()
            c.close()

        postgres_dbscan('cargo_tanker_position', new_table_name, eps , min_samples, 
                        mmsi_list, loc_conn)
        
        # add geom colum to the new tables
        make_tables_geom(new_table_name, loc_conn)
        
        loc_conn.close()

        #timekeeping
        tock = datetime.datetime.now()
        lapse = tock - tick
        print ('Time elapsed: {}'.format(lapse))

#%% Second round
make_tables_geom('dbscan_results_cargo_by_mmsi_10_50', loc_conn)
        
