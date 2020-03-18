#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:06:44 2019

@author: patrickmaus
"""
import psycopg2
import pandas as pd

#choose db.  this is used for connections throughout the script
database = 'ais_test'

#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database=database)
c = conn.cursor()
if c:
    print('Connection to {} is good.'.format(database))
else:
    print('Error connecting.')
c.close()

#%% Function for executing SQL
def execute_sql(SQL):
    c = conn.cursor()
    c.execute(SQL)
    conn.commit()
    c.close()
#%% Select random samples
mmsi_sample = pd.read_csv('/Users/patrickmaus/Documents/projects/AIS_project/ingest_pipeline/sample_mmsi.csv')
mmsi_list = mmsi_sample['mmsi'].astype('str').to_list()
mmsi_tuple = tuple(mmsi_list)

#%% make ship_trips_sample table

c = conn.cursor()
c.execute("""CREATE TABLE ship_trips_sample AS
          SELECT * FROM ship_trips WHERE mmsi IN %s""", (mmsi_tuple,))
conn.commit()
c.close()

#%% make ship_position_sample table
c = conn.cursor()
c.execute("""CREATE TABLE ship_position_sample AS
          SELECT * FROM ship_position WHERE mmsi IN %s""", (mmsi_tuple,))
conn.commit()
c.close()

#%% Make indices
c = conn.cursor()

c.execute("""CREATE INDEX ship_position_sample_mmsi_idx on 
          ship_position_sample (mmsi);""")
conn.commit()
c.execute("""CREATE INDEX ship_position_sample_geog_idx 
          ON ship_position_sample USING GIST (geog);""")
conn.commit()
c.execute("""CREATE INDEX ship_trips_sample_mmsi_idx 
          on ship_trips_sample (mmsi);""")
conn.commit()

c.close()

#%% dedupe table
def dedupe_table(table):
    c = conn.cursor()
    c.execute("""CREATE TABLE tmp as 
          (SELECT * from (SELECT DISTINCT * FROM {}) as t);""".format(table))
    c.execute("""DELETE from {};""".format(table))
    c.execute("""INSERT INTO {} SELECT * from tmp;""".format(table))
    c.execute("""DROP TABLE tmp;""")
    conn.commit()
    c.close()
#%%
dedupe_table('ship_position_sample')
