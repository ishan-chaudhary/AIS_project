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

#%%
import aws_credentials as a_c
user = a_c.user
host = a_c.host
port = '5432'
database = 'aws_ais_clustering'
password = a_c.password

aws_conn = psycopg2.connect(host=host,database=database, user=user,password=password)
aws_c = aws_conn.cursor()
if aws_c:
    print('Connection to AWS is good.'.format(database))
else: print('Connection failed.')
aws_c.close()


# def create_aws_engine(database):
#     import aws_credentials as a_c
#     user = a_c.user
#     host = a_c.host
#     port = '5432'
#     password = a_c.password
#     try:
#         aws_engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, database))
#         print('AWS Engine created and connected.')
#         return aws_engine
#     except:
#         print('AWS Engine creation failed.')
#         return None

# aws_engine = create_aws_engine('aws_ais_clustering')

#%%
database='ais_test'
loc_conn = psycopg2.connect(host="localhost",database=database)
c = loc_conn.cursor()
if c:
    print('Connection to {} is good.'.format(database))
else:
    print('Error connecting.')
c.close()

    # # drop table if an old one exists
    # c = conn.cursor()
    # c.execute("""DROP TABLE IF EXISTS {}""".format(new_table_name))
    # conn.commit()
    # c.close()

#%% Get mmsis to cluster

# all mmsis
source_table = 'ship_trips'
c = loc_conn.cursor()
c.execute("""SELECT DISTINCT(mmsi) FROM {};""".format(source_table))
mmsi_list = c.fetchall()
c.close()
#%%
# all tankers and cargo ships
c = loc_conn.cursor()
c.execute("""SELECT DISTINCT(mmsi)
          FROM cargo_tanker_mmsis;""")
cargo_tanker_mmsi_list = c.fetchall()
c.close()

cargo_tanker_mmsi_sample = random.sample(cargo_tanker_mmsi_list,200)


#%%
def postgres_dbscan(source_table, new_table_name, eps_km, min_samples, 
                    mmsi_list, conn):
    
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

    print("""Starting processing on DBSCAN with eps_km={} and
          min_samples={} """.format(str(eps_km), str(min_samples)))
    
    for mmsi in mmsi_list:          
        dbscan_sql = """INSERT INTO {} (id, mmsi, lat, lon, clust_id)
        SELECT id, mmsi, lat, lon,
        ST_ClusterDBSCAN(Geometry(geog), eps := {}, minpoints := {})
        over () as clust_id
        FROM {}
        WHERE mmsi = '{}';""".format(new_table_name, str(eps), str(min_samples), source_table, mmsi[0])
        
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

print('Function run at:', datetime.datetime.now())
epsilons = [2, 5, 7, 10]
samples = [50, 100, 250, 500]

for eps_km in epsilons:
    for min_samples in samples:

        loc_conn = psycopg2.connect(host="localhost",database=database)
        tick = datetime.datetime.now()
        
        #this formulation will yield epsilon based on km desired
        kms_per_radian = 6371.0088
        eps = eps_km / kms_per_radian
    
        # make the new table name
        new_table_name = ('dbscan_results_cargo_by_mmsi_' + str(eps_km).replace('.','_') +
                          '_' + str(min_samples))
        
        # drop table if exists if needed.
        c = loc_conn.cursor()
        c.execute("""DROP TABLE IF EXISTS {}""".format(new_table_name))
        loc_conn.commit()
        c.close()
        
        # pass the epsilon in km.  the function will convert it to radians
        postgres_dbscan('cargo_tanker_position', new_table_name, eps_km , min_samples, 
                        cargo_tanker_mmsi_sample, loc_conn)
        
        # add geom colum to the new tables
        make_tables_geom(new_table_name, loc_conn)

        #timekeeping
        tock = datetime.datetime.now()
        lapse = tock - tick
        print ('Time elapsed: {}'.format(lapse))
        
        
