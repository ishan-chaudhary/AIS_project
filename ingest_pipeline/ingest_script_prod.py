#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:44:11 2019

@author: patrickmaus
"""

#db connections
import psycopg2
from sqlalchemy import create_engine

#time tracking
import datetime

#file management
import glob

# for scraping, unzipping, and more file management
#from zipfile import ZipFile
#import shutil
#import os

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

aws_conn = gsta.connect_psycopg2(gsta_config.aws_ais_cluster_params)
loc_conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
aws_conn.close()    
loc_conn.close()


#%% drop other tables
# Keep commented out unless we are re-ingesting everything.
#gsta.drop_table('ship_info')
#gsta.drop_table('ship_position')
#gsta.drop_table('imported_ais')
#gsta.drop_table('ship_trips')
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
    log.write('Total Time elapsed: {} \n'.format(lapse))
    log.close()
    
    print('Total Time elapsed: ', lapse)

#%% create an imported_ais table to hold each file as its read in
c = conn.cursor()
c.execute("""CREATE TABLE imported_ais (
  	mmsi 			text,
    time     		timestamp,
	lat				numeric,
	lon				numeric,
	sog				varchar,
	cog				varchar,
	heading			varchar,
	ship_name		text,
	imo				varchar,
	callsign 		varchar,
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
c.execute("""CREATE TABLE IF NOT EXISTS cargo_ship_position
(
    mmsi text,
    time timestamp,
    lat numeric,
    lon numeric
);""")
conn.commit()

c.execute("""CREATE INDEX ship_position_mmsi_idx on ship_position (mmsi);""")
conn.commit()
c.execute("""CREATE INDEX ship_position_geom_idx 
          ON ship_position USING GIST (geom);""")
conn.commit()
c.close()

#%% create WPI table funtion

wpi_csv_path = '/Users/patrickmaus/Documents/projects/AIS_project/WPI_data/wpi_clean.csv'

def make_wpi(conn, wpi_csv_path=wpi_csv_path):
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
    c.execute("""CREATE INDEX wpi_geog_idx
              ON wpi
              USING GIST (geog);""")
    conn.commit()
    c.execute("""COPY wpi FROM '{}'
        WITH (format csv, header);""".format(wpi_csv_path))
    conn.commit()
    c.close()
    print('WPI created')
    
loc_cargo_conn = connect_psycopg2(loc_cargo_params)
make_wpi(conn=loc_cargo_conn)
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

## Need to add ship type here.  Right now its added manually
# alter table ship_trips 
# add column ship_type text

# --couldnt get this to work
# insert into ship_trips (ship_type)
# values (select ship_info.ship_type
# from ship_info
# join ship_trips
# on ship_trips.mmsi = ship_info.mmsi)

# --make new summary table
# create table ship_summary as
# select ship_info.mmsi, ship_info.ship_type,
# ship_trips.position_count, ship_trips.line_length_km, ship_trips.time_diff
# from ship_info, ship_trips
# where ship_trips.mmsi = ship_info.mmsi

def make_ship_trips():
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS ship_trips;""")
    conn.commit()
    c.execute("""CREATE TABLE ship_trips AS
    SELECT 
        id,   
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

#file_name = '/Users/patrickmaus/Documents/projects/AIS_data/2017/AIS_2017_01_Zone09.csv'

for file_name in glob.glob(source_dir):
   tick = datetime.datetime.now()
   print ('Started file {} at {} \n'.format(file_name[-22:], tick.time()))
   
   function_tracker(parse_ais_SQL(file_name), 'parse original AIS data')
   
   tock = datetime.datetime.now()
   lapse = tock - tick
   print ('Time elapsed: {} \n'.format(lapse))
   
   log = open(log_name, 'a+')
   log.write('Starting file {} at {} \n'.format(file_name[-22:], tick.time()))
   log.write('Time elapsed: {} \n'.format(lapse))
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
