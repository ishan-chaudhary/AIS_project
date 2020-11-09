#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:06:44 2019

@author: patrickmaus
"""
import psycopg2
import pandas as pd
import random

# Geo-Spatial Temporal Analysis package
import gsta
import db_config

aws_conn = gsta.connect_psycopg2(db_config.aws_ais_cluster_params)
loc_conn = gsta.connect_psycopg2(db_config.loc_cargo_params)
aws_conn.close()    
loc_conn.close()

conn = gsta.connect_psycopg2(db_config.loc_cargo_params)
#%% Function for executing SQL
def execute_sql(SQL):
    c = conn.cursor()
    c.execute(SQL)
    conn.commit()
    c.close()
#%% Select random samples
c = conn.cursor()
c.execute('select distinct(mmsi) from cargo_ship_position')
cargo_mmsi_list = c.fetchall()
cargo_mmsi_sample = random.sample(cargo_mmsi_list,28)
mmsi_tuple = tuple(cargo_mmsi_sample)

# mmsi_sample = pd.read_csv('/Users/patrickmaus/Documents/projects/AIS_project/ingest_pipeline/sample_mmsi.csv')
# mmsi_list = mmsi_sample['mmsi'].astype('str').to_list()
# mmsi_tuple = tuple(mmsi_list)

#%% make ship_trips_sample table

c = conn.cursor()
c.execute("""CREATE TABLE ship_trips_sample AS
          SELECT * FROM ship_trips WHERE mmsi IN %s""", (mmsi_tuple,))
conn.commit()
c.close()

#%% make ship_position_sample table
c = conn.cursor()
c.execute("""CREATE TABLE ship_position_sample AS
          SELECT * FROM cargo_ship_position WHERE mmsi IN %s""", (mmsi_tuple,))
conn.commit()
c.close()

#%% Make indices
c = conn.cursor()

c.execute("""CREATE INDEX ship_position_sample_mmsi_idx on 
          ship_position_sample (mmsi);""")
conn.commit()
c.execute("""CREATE INDEX ship_position_sample_geog_idx 
          ON ship_position_sample USING GIST (geom);""")
conn.commit()
c.execute("""CREATE INDEX ship_trips_sample_mmsi_idx 
          on ship_trips_sample (mmsi);""")
conn.commit()

c.close()

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


