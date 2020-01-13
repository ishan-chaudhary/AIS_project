#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:44:11 2019

@author: patrickmaus
"""

#db connections
import psycopg2
from sqlalchemy import create_engine

#parsing
import pandas as pd
#time tracking
import datetime

#file management
import os
import glob
from zipfile import ZipFile
import shutil

#choose db.  this is used for connections throughout the script
database = 'ais_test'

#%% Make and test conn and cursor
tick = datetime.datetime.now()

conn = psycopg2.connect(host="localhost",database=database)
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()

tock = datetime.datetime.now()
lapse = tock - tick

print('Time elapsed: ', lapse)

#%% Create "ship_info" table in the "ais_data" database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_info
(
    mmsi text,
    name text,
    type text
);""")
conn.commit()

# Creat index on MMSI
c.execute("""CREATE INDEX ship_info_mmsi_idx on ship_info (mmsi);""")
conn.commit()
c.close()

#%% Create "ship_position" table in the  database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_position
(
    mmsi text,
    time timestamp,
    lat numeric,
    lon numeric
);""")
conn.commit()

c.execute("""CREATE INDEX ship_position_mmsi_idx on ship_position (mmsi);""")
conn.commit()
c.close()
#%% Drop tables if needed
def drop_table(table):
    c = conn.cursor()
    c.execute('drop table if exists {} cascade'.format(table))
    conn.commit()
    c.close()
    
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

def parse_ship_position(df):
    ship_position = df[['MMSI', 'BaseDateTime', 'LAT', 'LON']]
    ship_position = ship_position.rename({'MMSI':'mmsi', 'BaseDateTime':'time',
                                          'LAT':'lat', 'LON':'lon'}, axis='columns')
    ship_position.to_sql('ship_position', engine, if_exists='append', index=False)
    conn.commit()

def parse_ship_info(df):
    ship_info = df[['MMSI', 'VesselName', 'VesselType']]
    ship_info = ship_info.rename({'MMSI':'mmsi', 'VesselName':'name',
                                  'VesselType':'type'}, axis='columns')
    ship_info.drop_duplicates(inplace=True)
    ship_info.to_sql('ship_info', engine, if_exists='append',
                     index = False)
    conn.commit()

def parse_ais(file_name, chunk_size, engine):
    for df in pd.read_csv(file_name, chunksize = chunk_size):
        try:
            conn = psycopg2.connect(host="localhost",database=database)
            c = conn.cursor()
            # ship_position table first
            parse_ship_position(df)
            # then ship_info
            parse_ship_info(df)
            # close the conn and cursor
            conn.close()
            c.close()
        except:
            print('Error in index range for ship_info: ',
                  df.iloc[0].name, df.iloc[-1].name)
            c.close()
            conn.close()
#%%
drop_table('ship_info')
drop_table('ship_position')
#%%
source_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017/*.csv'
destination_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017_unzipped'
file_name = '/Users/patrickmaus/Documents/projects/AIS_data/2017/AIS_2017_01_Zone09.csv'
chunk_size = 100000
engine = create_engine('postgresql://patrickmaus@localhost:5432/{}'.format(database))



#for file_name in glob.glob(source_dir):
tick = datetime.datetime.now()
print('Starting: ', file_name, ' at time: ', tick)

#parse_ais(file_name, chunk_size, engine)
dedupe_table('ship_info')

tock = datetime.datetime.now()
lapse = tock - tick

print('Time elapsed: ', lapse)

#%% Processing log
#time_now = datetime.datetime.now()
#notes = 'run with mmsi and 10 mil chunk size'
#log = open('/Users/patrickmaus/Documents/projects/AIS_project/proc_log.txt', 'a+')
#log.write("""At {}, a run occurred with: {}.  It started at {}, finished at {},
#          for a total run time of {}.\n\n""".format(time_now, notes, tick, tock, lapse))
#log.close()

# Notes
    # Create a processing file to record start and stop times for each run.
    # Add filters to just get zones 14 through 20 so dont have to manually delete
    # add option for number of months to ingest so dont have to manually delete
    # add indexes

#%% v2 for zipped files 
#source_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017/*.zip'
#destination_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017_unzipped'
#
#chunk_size = 1000000
#engine = create_engine('postgresql://patrickmaus@localhost:5432/ais_data')
#
#for file_name in glob.glob(source_dir):
#
#    tick = datetime.datetime.now()
#
#    print('Starting file: ', file_name)
#    with ZipFile(file_name, 'r') as zip:
#        zip.extractall(destination_dir)
#
#
#    for df in pd.read_csv(file_name, chunksize = chunk_size):
#        try:
#            c = conn.cursor()
#            ship_info = df[['MMSI', 'VesselName', 'IMO', 'CallSign', 'VesselType']]
#            ship_info = ship_info.rename({'MMSI':'mmsi', 'VesselName':'name',
#                                          'IMO':'imo', 'CallSign':'callsign',
#                                          'VesselType':'type'}, axis='columns')
#            ship_info.set_index('mmsi', inplace=True)
#            ship_info.drop_duplicates(inplace=True)
#            ship_info.to_sql('ship_info', engine, if_exists='append',
#                             index = False)
#            conn.commit()
#            c.close()
#        except:
#            print('Error in index range for ship_info: ',
#                  df.iloc[0].name, df.iloc[-1].name)
#
#        try:
#            c = conn.cursor()
#            ship_position = df[['MMSI', 'BaseDateTime', 'LAT', 'LON']]
#            ship_position = ship_position.rename({'MMSI':'mmsi', 'BaseDateTime':'time',
#                                          'LAT':'lat', 'LON':'lon'}, axis='columns')
#            ship_position.to_sql('ship_position', engine, if_exists='append',
#                                 index=False)
#            conn.commit()
#            c.close()
#        except:
#            print('Error in index range for ship_position: ',
#                  df.iloc[0].name, df.iloc[-1].name)
#
#    c = conn.cursor()
#    c.execute("""DELETE FROM ship_info WHERE ship_info.id NOT IN
#                  (SELECT id FROM
#                  (SELECT DISTINCT ON (mmsi) *
#                  FROM ship_info) as foo);""")
#    conn.commit()
#
#    c.close()
#
#    os.remove(file_name)
#    shutil.rmtree(destination_dir)
#    tock = datetime.datetime.now()
#    lapse = tock - tick
#
#    print('Finished: ', file_name)
#    print('Time elapsed: ', lapse)
#
#    #%%
#c = conn.cursor()
#c.execute("""create table ship_trips_3 AS
#SELECT * FROM ship_trips;;""")
#conn.commit()
#c.close()
