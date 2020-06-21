#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:44:11 2019
@author: patrickmaus
"""

# time tracking
import datetime

# file management
import glob

# scrape
import requests
import zipfile
import os

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config


#%%
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_full_params)

# %% drop other tables
# Keep commented out unless we are re-ingesting everything.
gsta.drop_table('ship_info', conn)
gsta.drop_table('cargo_ship_position', conn)
gsta.drop_table('imported_ais', conn)
gsta.drop_table('ship_trips', conn)
# %% start processing
first_tick = datetime.datetime.now()
first_tick_pretty = first_tick.strftime("%Y_%m_%d_%H%M")
log_name = '/Users/patrickmaus/Documents/projects/AIS_project/proc_logs/proc_log_{}.txt'.format(first_tick_pretty)
log = open(log_name, 'a+')
log.write('Starting processing at: {} \n'.format(first_tick_pretty))
log.close()
print('Starting processing at: ', first_tick_pretty)


# %% create a function to print and log milestones
def function_tracker(function, function_name,
                     tick_now=datetime.datetime.now()):
    print('Starting function {}'.format(function_name))
    function
    tock_now = datetime.datetime.now()
    lapse = tock_now - tick_now

    log = open(log_name, 'a+')
    log.write('Starting function {} \n'.format(function_name))
    log.write('Total Time elapsed: {} \n'.format(lapse))
    log.close()

    print('Total Time elapsed: ', lapse)


# %% create an imported_ais table to hold each file as its read in
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
# %% Create "ship_info" table in the database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS ship_info
(
    mmsi text,
    ship_name text,
    ship_type text
);""")
conn.commit()

# Create index on MMSI
c.execute("""CREATE INDEX ship_info_mmsi_idx on ship_info (mmsi);""")
conn.commit()
c.close()

# %% Create "ship_position" table in the  database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS cargo_ship_position
(   id serial,
    mmsi text,
    time timestamp,
    geog geography,
    lat numeric,
    lon numeric
);""")
conn.commit()
c.close()

# %% create WPI table funtion

wpi_csv_path = '/Users/patrickmaus/Documents/projects/AIS_project/WPI_data/wpi_clean.csv'
def make_wpi(conn, wpi_csv_path=wpi_csv_path):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS wpi (
                    index_no	int,
                    region_no	int,
                    port_name	text,
                    country		text,
                    latitude	numeric,
                    longitude	numeric,
                    geog		geography,
                    geom		geometry);""")
    c.execute("""CREATE INDEX wpi_geog_idx
              ON wpi
              USING GIST (geog);""")
    conn.commit()
    c.execute("""COPY wpi FROM '{}'
        WITH (format csv, header);""".format(wpi_csv_path))
    conn.commit()
    c.close()
    print('WPI created')


#%%
def download_url(link, download_path, unzip_path, file_name, chunk_size=128):
    print('Testing link...')
    r = requests.get(link, stream=True)
    if r.status_code == 200:
        print('Link good for {}!'.format(file_name))
    else:
        print('Link did not return 200 status code')
    with open(download_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    print('File downloaded.')
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(path=unzip_path)
    print('File unzipped!')
    # delete the zipped file
    os.remove(download_path)
    print('Zip file deleted.')
    print()

# %% read in and parse the original file into new tables
def parse_ais_SQL(file_name):
    c = conn.cursor()
    c.execute("""COPY imported_ais FROM '{}'
                WITH (format csv, header);""".format(file_name))
    conn.commit()
    # this will only insert positions from cargo ship types
    c.execute("""INSERT INTO cargo_ship_position (mmsi, time, geog, lat, lon)
                SELECT mmsi, 
                time, 
                ST_SetSRID(ST_MakePoint(lon, lat), 4326), 
                lat, 
                lon 
                FROM imported_ais
                where ship_type IN (
                '70','71','72','73','74','75','76','77','78','79',
                '1003','1004','1016');""")
    conn.commit()
    c.execute("""INSERT INTO ship_info (mmsi, ship_name, ship_type)
                SELECT DISTINCT mmsi, ship_name, ship_type from imported_ais
                where ship_type IN ('70','71','72','73','74','75','76','77',
                '78','79', '1003','1004','1016');""")
    conn.commit()
    c.execute("""DELETE FROM imported_ais""")
    conn.commit()
    c.close()


# %% Populate ship_trips table from ship_postion table

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
                FROM cargo_ship_position as pos
                GROUP BY pos.mmsi) AS foo;""")
    conn.commit()
    c.execute("""CREATE INDEX ship_trips_mmsi_idx on ship_trips (mmsi);""")
    conn.commit()
    c.close()

# %% run all the functions using the function tracker
# set variables for functions
folder = '/Users/patrickmaus/Documents/projects/AIS_data'

year = 2017
for zone in [9, 10, 11, 14, 15, 16, 17, 18, 19, 20]:
    for month in range(1, 13):
        file_name = f'{str(year)}_{str(month).zfill(2)}_{str(zone).zfill(2)}'
        download_path = f'{folder}/{file_name}.zip'
        unzip_path = f'{folder}/{str(year)}'
        url_path = f'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{str(year)}/'
        url_file = f'AIS_{str(year)}_{str(month).zfill(2)}_Zone{str(zone).zfill(2)}'
        link = url_path + url_file + '.zip'

        tick = datetime.datetime.now()
        print('Started file {} at {} \n'.format(file_name[-22:], tick.time()))

        # use the download_url function to download the zip file, unzip it, and
        # delete the zip file
        download_url(link, download_path, unzip_path, file_name)

        print('Starting to parse raw data...')
        file_path = folder + '/' + str(year) + '/AIS_ASCII_by_UTM_Month/2017_v2/' + url_file + '.csv'
        function_tracker(parse_ais_SQL(file_path), 'parse original AIS data')
        print(f'Parsing complete for {file_name}.')

        # delete the csv file
        os.remove(file_path)
        print('CSV file deleted.')

        tock = datetime.datetime.now()
        lapse = tock - tick
        print('Time elapsed: {} \n'.format(lapse))

        log = open(log_name, 'a+')
        log.write('Starting file {} at {} \n'.format(file_name[-22:], tick.time()))
        log.write('Time elapsed: {} \n'.format(lapse))
        log.close()

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

#%% additional functions and index building

function_tracker(gsta.dedupe_table('ship_info', conn), 'dedupe ship_info')
function_tracker(make_ship_trips(), 'make_ship_trips')
#%%

loc_cargo_conn = gsta.connect_psycopg2(gsta_config.loc_cargo_full_params)
make_wpi(conn=loc_cargo_conn)
loc_cargo_conn.close()

c.execute("""CREATE INDEX ship_position_mmsi_idx 
            on cargo_ship_position (mmsi);""")
conn.commit()
c.execute("""CREATE INDEX ship_position_geom_idx 
            ON cargo_ship_position USING GIST (geom);""")
conn.commit()