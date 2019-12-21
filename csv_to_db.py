#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:44:11 2019

@author: patrickmaus
"""

#for connecting to databases
import psycopg2
from sqlalchemy import create_engine

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
#%%
df = pd.read_csv('../AIS_data/AIS_ASCII_by_UTM_Month/2017_v2/AIS_2017_01_Zone04.csv')

#%% Parse df into two seperate dataframes for the different tables
ship_info = df[['MMSI', 'VesselName', 'IMO', 'CallSign', 'VesselType',
                'Length', 'Width']]
ship_info.drop_duplicates(inplace=True)
ship_position = df[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG',
                    'Heading', 'Status']]
ship_info.drop_duplicates(inplace=True)
del df

#%%
conn = psycopg2.connect(host="localhost",database="ais_data")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()
#%% Drop tables if needed
c = conn.cursor()
c.execute('drop table if exists ship_info')
c.execute('drop table if exists ship_position')
conn.commit()
c.close()

#%%
# Create "ship_info" table in the "ais_data" database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_info
(
    MMSI text,
    name text,
    IMO text,
    callsign text,
    type text,
    length numeric,
    width numeric
);""")
conn.commit()
c.close()
#%% Insert ship_info data
c = conn.cursor()
sql_insert = """INSERT INTO ship_info(MMSI, name, IMO, callsign, type, length,
width) VALUES(%s,%s,%s,%s,%s,%s,%s)"""
for name,row in ship_info.iterrows():
    try:
        c.execute(sql_insert, (row[0], row[1], row[2], row[3], row[4], row[5], row[6]))
        conn.commit()
    except:
        print('Failure to insert:', name)
c.close()

#%%
# Create "ship_position" table in the "ais_data" database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_position
(
    MMSI text,
    time timestamp,
    lat numeric,
    lon numeric,
    sog numeric,
    cog numeric,
    heading numeric,
    status text
);""")
conn.commit()
c.close()
#%% Insert ship_position data
c = conn.cursor()
sql_insert = """INSERT INTO ship_position(MMSI, time, lat, lon, sog, cog,
heading, status) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)"""
for name,row in ship_position.iterrows():
    try:
        c.execute(sql_insert, (row[0], row[1], row[2], row[3], row[4], row[5],
                               row[6], row[7]))
        conn.commit()
    except:
        print('Failure to insert:', name)
c.close()
#%%
del ship_info
del ship_position
#%% Using SQL Alchemy
engine = create_engine('postgresql://patrickmaus@localhost:5432/ais_data')
ship_info.to_sql('ship_info', engine, if_exists='replace')
ship_position.to_sql('ship_position', engine, if_exists='replace')
