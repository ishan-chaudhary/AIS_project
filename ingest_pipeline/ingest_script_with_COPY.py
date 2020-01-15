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

###  


#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database=database)
c = conn.cursor()
if c:
    print('Connection to {} is good.'.format(database))
c.close()

#%% Drop tables if needed
def drop_table(table):
    c = conn.cursor()
    c.execute('drop table if exists {} cascade'.format(table))
    conn.commit()
    c.close()
#%% drop other tables
drop_table('ship_info')
drop_table('ship_position')
drop_table('imported_ais')
drop_table('ship_trips')
#%% start processing
first_tick = datetime.datetime.now()

log_name = '/Users/patrickmaus/Documents/projects/AIS_project/proc_logs/proc_log_{}.txt'.format(first_tick.time())
log = open(log_name, 'a+')
log.write('Starting processing at: {} \n'.format(first_tick.time()))
log.close()
print('Starting processing at: ', first_tick)

#%% create a function to print and log milestones
def function_tracker(function, function_name, 
                     tick_now = datetime.datetime.now()):
    print('Starting function {}'.format(function_name))
    function
    tock_now = datetime.datetime.now()
    lapse = tock_now - tick_now
    
    log = open(log_name, 'a+')
    log.write('Starting function {} \n'.format(function_name))
    log.write('Time elapsed: {} \n'.format(lapse))
    log.close()
    
    print('Time elapsed: ', lapse)
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

#%% create an imported_ais table to hold each file as its read in
c = conn.cursor()
c.execute("""CREATE TABLE imported_ais (
  	mmsi 			text,
	time		timestamp,
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
    geog geography,
    lat numeric,
    lon numeric
);""")
conn.commit()

c.execute("""CREATE INDEX ship_position_mmsi_idx on ship_position (mmsi);""")
conn.commit()
c.execute("""CREATE INDEX ship_position_geog_idx 
          ON ship_position USING GIST (geog);""")
conn.commit()
c.close()

#%% create WPI table funtion
def make_wpi(wpi_csv_path=wpi_csv_path):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS wpi (
        	index_no		int,
        	region_no	int,
        	port_name	text,
        	country		text,
        	latitude		numeric,
        	longitude	numeric,
        	geog			geography,
        	geom			geometry);""")
    conn.commit()
    c.execute("""COPY wpi FROM '{}'
        WITH (format csv, header);""".format(wpi_csv_path))
    conn.commit()
    c.close()
#%% read in and parse the original file into new tables
def parse_ais_SQL(file_name):
    c = conn.cursor()
    c.execute("""COPY imported_ais FROM '{}'
                WITH (format csv, header);""".format(file_name))
    conn.commit()
    c.execute("""INSERT INTO ship_position (mmsi, time, geog, lat, lon)
                SELECT mmsi, 
                time, 
                ST_SetSRID(ST_MakePoint(lon, lat), 4326), 
                lat, 
                lon 
                FROM imported_ais;""")
    conn.commit()
    c.execute("""INSERT INTO ship_info (mmsi, ship_name, ship_type)
                SELECT DISTINCT mmsi, ship_name, ship_type from imported_ais;""")
    conn.commit()
    c.execute("""DELETE FROM imported_ais""")
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
		last_date - first_date as time_diff,
        line
        FROM (
                SELECT pos.mmsi,
                COUNT (pos.geog) as position_count,
                ST_MakeLine((pos.geog::geometry) ORDER BY pos.time) AS line,
                MIN (pos.time) as first_date,
                MAX (pos.time) as last_date
                FROM ship_position as pos
                GROUP BY pos.mmsi) AS foo;""")
    conn.commit()
    c.execute("""CREATE INDEX ship_trips_mmsi_idx on ship_trips (mmsi);""")
    conn.commit()
    c.close()

#%% run all the functions using the function tracker
    
source_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017/*.csv'
wpi_csv_path = '/Users/patrickmaus/Documents/projects/AIS_project/WPI_data/wpi_clean.csv'

#file_name = '/Users/patrickmaus/Documents/projects/AIS_data/2017/AIS_2017_01_Zone09.csv'

for file_name in glob.glob(source_dir):
   tick = datetime.datetime.now()
   print ('Started file {} at {} \n'.format(file_name[-22:], tick.time()))
   
   function_tracker(parse_ais_SQL(file_name), 'parse original AIS data')
   
#   tock = datetime.datetime.now()
#   lapse = tock - tick
#   print ('Time elapsed: {} \n'.format(lapse))
   
   log = open(log_name, 'a+')
   log.write('Starting file {} at {} \n'.format(file_name[-22:], tick.time()))
#   log.write('Time elapsed: {} \n'.format(lapse))
   log.close()
   
function_tracker(dedupe_table('ship_info'), 'dedupe ship_info')
function_tracker(make_ship_trips(), 'make_ship_trips')  
function_tracker(make_wpi(wpi_csv_path), 'make_wpi')

#%%
last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

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

#destination_dir = '/Users/patrickmaus/Documents/projects/AIS_data/2017_unzipped'
