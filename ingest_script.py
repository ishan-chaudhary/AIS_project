#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:44:11 2019

@author: patrickmaus
"""

import psycopg2
import pandas as pd
from sqlalchemy import create_engine
import os
import glob
from zipfile import ZipFile 
import shutil
import datetime

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
    mmsi text,
    name text,
    imo text,
    callsign text,
    type text
);""")
conn.commit()
c.execute("""CREATE INDEX mmsi_index on ship_info (mmsi);""")
conn.commit()
c.close()
#%% Create "ship_position" table in the "ais_data" database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_position
(  
    mmsi text,
    time timestamp,
    lat numeric,
    lon numeric
);""")
conn.commit()
#c.execute("""CREATE INDEX mmsi_index_pos on ship_position (mmsi);""")
#conn.commit()
#c.execute("""CREATE INDEX time_index_pos on ship_position (time);""")
#conn.commit()
c.close()

#%% v1
source_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017/*.csv'
destination_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017_unzipped'

chunk_size = 100000
engine = create_engine('postgresql://patrickmaus@localhost:5432/ais_data')
conn.close()
for file_name in glob.glob(source_dir):
    
    tick = datetime.datetime.now()
    print('Starting: ', file_name, ' at time: ', tick)
       
    for df in pd.read_csv(file_name, chunksize = chunk_size):          
        try:
            conn = psycopg2.connect(host="localhost",database="ais_data")
            c = conn.cursor()
            
            ship_position = df[['MMSI', 'BaseDateTime', 'LAT', 'LON']]
            ship_position = ship_position.rename({'MMSI':'mmsi', 'BaseDateTime':'time', 
                                          'LAT':'lat', 'LON':'lon'}, axis='columns')
            ship_position.to_sql('ship_position', engine, if_exists='append',
                                 index=False)
            conn.commit()
            c.close()
            conn.close()
        except:
            print('Error in index range for ship_position: ', 
                  df.iloc[0].name, df.iloc[-1].name)

    
    #os.remove(file_name)
    #shutil.rmtree(destination_dir)
    tock = datetime.datetime.now()
    lapse = tock - tick
    
    print('Finished: ', file_name, ' at time: ', tock)
    print('Time elapsed: ', lapse)
    
#%% Processing log
time_now = datetime.datetime.now()
notes = 'run with mmsi and 10 mil chunk size'

log = open('/Users/patrickmaus/Documents/projects/AIS_project/proc_log.txt', 'a+')
log.write("""At {}, a run occurred with: {}.  It started at {}, finished at {}, 
          for a total run time of {}.\n\n""".format(time_now, notes, tick, tock, lapse))
log.close()
#%%
# Notes
    # Create a processing file to record start and stop times for each run.
    # Add filters to just get zones 14 through 20 so dont have to manually delete
    # add option for number of months to ingest so dont have to manually delete
    # add indexes
    
#%% v2
source_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017/*.zip'
destination_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017_unzipped'

chunk_size = 1000000
engine = create_engine('postgresql://patrickmaus@localhost:5432/ais_data')

for file_name in glob.glob(source_dir):
    
    tick = datetime.datetime.now()
    
    print('Starting file: ', file_name)
    with ZipFile(file_name, 'r') as zip: 
        zip.extractall(destination_dir) 
    
    
    for df in pd.read_csv(file_name, chunksize = chunk_size):
        try:
            c = conn.cursor()
            ship_info = df[['MMSI', 'VesselName', 'IMO', 'CallSign', 'VesselType']]
            ship_info = ship_info.rename({'MMSI':'mmsi', 'VesselName':'name', 
                                          'IMO':'imo', 'CallSign':'callsign', 
                                          'VesselType':'type'}, axis='columns')
            ship_info.set_index('mmsi', inplace=True)
            ship_info.drop_duplicates(inplace=True)
            ship_info.to_sql('ship_info', engine, if_exists='append', 
                             index = False)
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
    
    os.remove(file_name)
    shutil.rmtree(destination_dir)
    tock = datetime.datetime.now()
    lapse = tock - tick
    
    print('Finished: ', file_name)
    print('Time elapsed: ', lapse)

    