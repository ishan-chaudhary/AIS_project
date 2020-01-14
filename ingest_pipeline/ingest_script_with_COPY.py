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
import glob
from zipfile import ZipFile
import shutil
import os

#choose db.  this is used for connections throughout the script
database = 'ais_test'
print('Starting processing at: ', datetime.datetime.now().time())
#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database=database)
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()


#%% Drop tables if needed
def drop_table(table):
    c = conn.cursor()
    c.execute('drop table if exists {} cascade'.format(table))
    conn.commit()
    c.close()
#%%
drop_table('ship_info')
drop_table('ship_position')
drop_table('imported_ais')
#%% create an imported_ais table to hold each file as its read in
c = conn.cursor()
c.execute("""CREATE TABLE imported_ais (
  	mmsi 			text,
	time			timestamp,
	lat				numeric,
	lon				numeric,
	sog				varchar,
	cog				varchar,
	heading			varchar,
	ship_name		text,
	imo				varchar,
	callsign		varchar,
	ship_type		text,
	status			varchar,
	len				varchar,
	width			varchar,
	draft			varchar,
	cargo			varchar);""")
conn.commit()
#%% Create "ship_info" table in the "ais_data" database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_info
(
    mmsi text,
    ship_name text,
    ship_type text
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
#%% read in and parse the original file into new tables

def parse_ais_SQL(file_name):
    c = conn.cursor()
    c.execute("""COPY imported_ais FROM '{}'
                WITH (format csv, header);""".format(file_name))
    conn.commit()
    c.execute("""INSERT INTO ship_position (mmsi, time, lat, lon)
                SELECT mmsi, time, lat, lon FROM imported_ais;""")
    conn.commit()
    c.execute("""INSERT INTO ship_info (mmsi, ship_name, ship_type)
                SELECT DISTINCT mmsi, ship_name, ship_type from imported_ais;""")
    conn.commit()
    c.execute("""DELETE FROM imported_ais""")
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
source_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017/*.csv'
#destination_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017_unzipped'
file_name = '/Users/patrickmaus/Documents/projects/AIS_data/2017/AIS_2017_01_Zone09.csv'

#%% populate the ship_position and ship_info table
#for file_name in glob.glob(source_dir):
tick = datetime.datetime.now()
print('Starting parse_ais_SQL: ', file_name[-22:])

parse_ais_SQL(file_name)

tock = datetime.datetime.now()
lapse = tock - tick
print('Time elapsed: ', lapse)

#%%
tick = datetime.datetime.now()
print('Starting dedupe_table(ship_info): ', file_name[-22:])

dedupe_table('ship_info')

tock = datetime.datetime.now()
lapse = tock - tick
print('Time elapsed: ', lapse)
#%% This function updates the geog, builds index on geog, and vaccums
def update_geog():
    c = conn.cursor()
    c.execute("""ALTER TABLE ship_position 
              ADD COLUMN geog geography (Point, 4326);""")
    conn.commit()
    c.execute("""UPDATE ship_position SET geog = ST_SetSRID(
            ST_MakePoint(lon, lat), 4326);""")
    conn.commit()
    c.execute("""CREATE INDEX ship_position_geog_idx 
              ON ship_position USING GIST (geog);""")
    conn.commit()
    c.close()

#%% Populate ship_trips table from ship_postion table
def make_ship_trips():
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS ship_trips;""")
    conn.commit()
    c.execute("""CREATE TABLE ship_trips AS
    SELECT 
        mmsi,
        	position_count,
		ST_Length(geography(line))/1000 AS line_length_km,
		first_date,
		last_date,
		last_date - first_date as time_diff
        FROM (
                SELECT pos.mmsi,
                COUNT (pos.geog) as position_count,
                ST_MakeLine(pos.geog ORDER BY pos.time) AS line,
                MIN (pos.time) as first_date,
                MAX (pos.time) as last_date
                FROM ship_position as pos
                GROUP BY pos.mmsi) AS foo;""")
    conn.commit()
    c.execute("""CREATE INDEX ship_trips_mmsi_idx on ship_trips (mmsi);""")
    conn.commit()
    c.close()
#%%
tick = datetime.datetime.now()
print('Starting update_geog', file_name[-22:])

update_geog()

tock = datetime.datetime.now()
lapse = tock - tick
print('Time elapsed: ', lapse)

#%%
tick = datetime.datetime.now()
print('Starting make_ship_trips: ', file_name[-22:])

make_ship_trips()

tock = datetime.datetime.now()
lapse = tock - tick
print('Time elapsed: ', lapse)

#%% Processing 
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

