#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:06:44 2019

@author: patrickmaus
"""

import psycopg2
import pandas as pd

#%% Establish connection and test
conn = psycopg2.connect(host="localhost",database="ais_data")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()
#%% Select random samples
mmsi_sample = pd.read_csv('/Users/patrickmaus/Documents/projects/AIS_project/sample_mmsi.csv')

#%% make sample table
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_position_sample
(  
    mmsi text,
    time timestamp,
    lat numeric,
    lon numeric,
    point_geog geometry
);""")
conn.commit()
c.close()
#%% Select all data for each mmsi and make a new table with sample data
for m in mmsi_sample['mmsi'].to_list():
    
    print('Getting records for MMSI {}...'.format(str(m)))
    c = conn.cursor()
    c.execute("""insert into ship_position_sample
              select * from ship_position 
              where mmsi = '{}'""".format(str(m)))
    conn.commit()
    c.close()
    print('{} added'.format(str(m)))

