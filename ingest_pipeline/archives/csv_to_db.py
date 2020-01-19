#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:44:11 2019

@author: patrickmaus
"""

#for connecting to databases
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

pd.set_option('display.expand_frame_repr', False)

#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database="ais_data")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()
#%% Drop tables if needed
c = conn.cursor()
c.execute('drop table if exists ship_info cascade')
c.execute('drop table if exists ship_position cascade')
conn.commit()
c.close()

#%% Create "ship_info" table in the "ais_data" database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_info
(
    id serial,
    mmsi text,
    name text,
    imo text,
    callsign text,
    type text
);""")
conn.commit()
c.close()
#%% Create "ship_position" table in the "ais_data" database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_position
(
    id serial,    
    mmsi text,
    time timestamp,
    lat numeric,
    lon numeric
);""")
conn.commit()
c.close()
#%% Function draft
file_path = '/Users/patrickmaus/Documents/projects/AIS_project/AIS_ASCII_by_UTM_Month/2017_v2/AIS_2017_01_Zone10.csv'
chunk_size = 1000000
engine = create_engine('postgresql://patrickmaus@localhost:5432/ais_data')

for df in pd.read_csv(file_path, chunksize = chunk_size):
    try:
        c = conn.cursor()
        ship_info = df[['MMSI', 'VesselName', 'IMO', 'CallSign', 'VesselType']]
        ship_info = ship_info.rename({'MMSI':'mmsi', 'VesselName':'name', 
                                      'IMO':'imo', 'CallSign':'callsign', 
                                      'VesselType':'type'}, axis='columns')
        ship_info.set_index('mmsi', inplace=True)
        ship_info.drop_duplicates(inplace=True)
        ship_info.to_sql('ship_info', engine, if_exists='append')
        conn.commit()
        c.close()
    except:
        print('Error in index range for ship_info: ', 
              df.iloc[0].name, df.iloc[-1].name)
        
    try:
        c = conn.cursor()
        ship_position = df[['MMSI', 'BaseDateTime', 'LAT', 'LON']]
        ship_position = ship_position.rename({'MMSI':'mmsi', 'BaseDateTime':'time', 
                                      'LAT':'lat', 'LON':'lon'}, axis='columns')
        ship_position.to_sql('ship_position', engine, if_exists='append',
                             index=False)
        conn.commit()
        c.close()
    except:
        print('Error in index range for ship_position: ', 
              df.iloc[0].name, df.iloc[-1].name)
          
c = conn.cursor()
c.execute("""DELETE FROM ship_info WHERE ship_info.id NOT IN 
              (SELECT id FROM 
              (SELECT DISTINCT ON (mmsi) *
              FROM ship_info) as foo);""")
conn.commit()

c.close()
    
#%%
ship_position = df[['MMSI', 'BaseDateTime', 'LAT', 'LON']]
ship_position = ship_position.rename({'MMSI':'mmsi', 'BaseDateTime':'time', 
                              'LAT':'lat', 'LON':'lon'}, axis='columns')
ship_position.to_sql('ship_position', engine, if_exists='append', index = False)
#%%
df.iloc[-1].name
df.iloc[0].name








#%% Insert ship_info data
c = conn.cursor()
sql_insert = """INSERT INTO ship_info(MMSI, name, IMO, callsign, type) VALUES(%s,%s,%s,%s,%s)"""
for name,row in ship_info.iterrows():
    try:
        c.execute(sql_insert, (row[0], row[1], row[2], row[3], row[4]))
        conn.commit()
    except:
        print('Failure to insert:', name)
c.close()


#%% Insert ship_position data
c = conn.cursor()
sql_insert = """INSERT INTO ship_position(mmsi, time, lat, lon, status) VALUES(%s,%s,%s,%s,%s)"""
for name,row in ship_position.iterrows():
    c.execute(sql_insert, (row[0], row[1], row[2], row[3], row[4]))
    conn.commit()

c.close()
#%%
del ship_info
del ship_position
#%% Using SQL Alchemy
#engine = create_engine('postgresql://patrickmaus@localhost:5432/ais_data')
#ship_info.to_sql('ship_info', engine, if_exists='replace')
#ship_position.to_sql('ship_position', engine, if_exists='replace')
