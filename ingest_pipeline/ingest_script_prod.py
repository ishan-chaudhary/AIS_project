#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:44:11 2019
@author: patrickmaus
"""
# time tracking
import datetime

# scrape
import requests
import zipfile
import os

# folder management
import glob

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

#%%
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_full_params)
# %% drop other tables
# # Keep commented out unless we are re-ingesting everything.
# gsta.drop_table('ship_info', conn)
# gsta.drop_table('cargo_ship_position', conn)
# gsta.drop_table('imported_ais', conn)
# gsta.drop_table('ship_trips', conn)
# %% start processing
first_tick = datetime.datetime.now()
first_tick_pretty = first_tick.strftime("%Y_%m_%d_%H%M")
log_name = '/Users/patrickmaus/Documents/projects/AIS_project/proc_logs/proc_log_{}.txt'.format(first_tick_pretty)
log = open(log_name, 'a+')
log.write('Starting processing at: {} \n'.format(first_tick_pretty))
log.close()
print('Starting processing at: ', first_tick_pretty)

# %% create an imported_ais table to hold each file as its read in
c = conn.cursor()
c.execute("""CREATE TABLE imported_ais (
  	uid 			text,
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
    uid text,
    ship_name text,
    ship_type text
);""")
conn.commit()

# Create index on uid
c.execute("""CREATE INDEX ship_info_uid_idx on ship_info (uid);""")
conn.commit()
c.close()

# %% Create "ship_position" table in the  database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS cargo_ship_position
(   id serial,
    uid text,
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

# %% read in and parse the original file into new tables
def parse_ais_SQL(file_name, conn=conn):
    c = conn.cursor()
    c.execute("""COPY imported_ais FROM '{}'
                WITH (format csv, header);""".format(file_name))
    conn.commit()
    # this will only insert positions from cargo ship types
    c.execute("""INSERT INTO cargo_ship_position (uid, time, geog, lat, lon)
                SELECT uid, 
                time, 
                ST_SetSRID(ST_MakePoint(lon, lat), 4326), 
                lat, 
                lon 
                FROM imported_ais
                where ship_type IN (
                '70','71','72','73','74','75','76','77','78','79',
                '1003','1004','1016');""")
    conn.commit()
    c.execute("""INSERT INTO ship_info (uid, ship_name, ship_type)
                SELECT DISTINCT uid, ship_name, ship_type from imported_ais
                where ship_type IN ('70','71','72','73','74','75','76','77',
                '78','79', '1003','1004','1016');""")
    conn.commit()
    c.execute("""DELETE FROM imported_ais""")
    conn.commit()
    c.close()

# %% Populate ship_trips table from ship position table
def make_ship_trips(new_table_name, conn):
    c = conn.cursor()
    c.execute(f"""DROP TABLE IF EXISTS {new_table_name};""")
    conn.commit()
    c.execute(f"""CREATE TABLE {new_table_name} AS
    SELECT 
        uid,
        position_count,
		ST_Length(geography(line))/1000 AS line_length_km,
		first_date,
		last_date,
		last_date - first_date as time_diff,
        line
        FROM (
                SELECT pos.uid,
                COUNT (pos.geog) as position_count,
                ST_MakeLine((pos.geog::geometry) ORDER BY pos.time) AS line,
                MIN (pos.time) as first_date,
                MAX (pos.time) as last_date
                FROM cargo_ship_position as pos
                GROUP BY pos.uid) AS foo;""")
    conn.commit()
    c.execute(f"""CREATE INDEX ship_trips_uid_idx on {new_table_name} (uid);""")
    conn.commit()
    c.close()

# %% run all the functions using the function tracker
# set variables for functions
folder = '/Users/patrickmaus/Documents/projects/AIS_data'
year = 2017
completed_files = []
error_files = []
#%%
# clear out any csvs first that are in the directory.
# since the files unzip to different subfolders, we have to use the blunt glob.glob approach
# to find the new csv.  Therefore, we are going to nuke any csvs in the directory or any subdirectories
print('Removing any .csv files in the target folder.')
for file in (glob.glob((folder + '/**/*.csv'), recursive=True)):
    os.remove(file)
    print('removed file', file[-22:])

# iterate through each month and each zone.  get the file and then parse them.
# need to add a check that writes each file and counts to a table for tracking iterations
for month in range(1, 13):
    for zone in [9, 10, 11, 14, 15, 16, 17, 18, 19, 20]:
        try:
            file_name = f'{str(year)}_{str(month).zfill(2)}_{str(zone).zfill(2)}'
            download_path = f'{folder}/{file_name}.zip'
            unzip_path = f'{folder}/{str(year)}'
            url_path = f'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{str(year)}/'
            url_file = f'AIS_{str(year)}_{str(month).zfill(2)}_Zone{str(zone).zfill(2)}'
            link = url_path + url_file + '.zip'

            tick = datetime.datetime.now()
            print('Started file {} at {}.'.format(file_name[-22:], tick.time()))

            if url_file in completed_files:
                print('File has already been processed.  Skipping.')
                continue
            elif url_file in error_files:
                print('File has been processed but errored out.  Skipping.')
                continue
            else:
                # use the download_url function to download the zip file, unzip it, and
                # delete the zip file
                download_url(link, download_path, unzip_path, file_name)

                print('Starting to parse raw data...')
                #file_path = folder + '/' + str(year) + '/AIS_ASCII_by_UTM_Month/2017_v2/' + url_file + '.csv'
                if len((glob.glob((folder + '/**/*.csv'), recursive=True))) == 1:
                    file = glob.glob((folder + '/**/*.csv'), recursive=True)[0]
                    parse_ais_SQL(file)
                    print(f'Parsing complete for {file_name}.')
                    # delete the csv file
                    os.remove(file)
                    print('CSV file deleted.')
                else:
                    print('More than one file expected.  Removing all of them.')
                    for file in (glob.glob((folder + '/**/*.csv'), recursive=True)):
                        os.remove(file)
                #append the file to the completed file list.
                completed_files.append(url_file)
        except:
            print('Error.  File name added to the error list.')
            error_files.append(url_file)
            log = open(log_name, 'a+')
            log.write('Error for file {}'.format(file_name[-22:]))
            log.close()

        # wrap up time keeping and logging
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
gsta.dedupe_table('ship_info', conn=loc_cargo_conn)
make_ship_trips('ship_trips', conn=loc_cargo_conn)
make_wpi(conn=loc_cargo_conn)
#%%

loc_cargo_conn = gsta.connect_psycopg2(gsta_config.loc_cargo_full_params)
make_ship_trips('ship_trips', conn=loc_cargo_conn)

c = loc_cargo_conn.cursor()
c.execute("""CREATE INDEX ship_position_uid_idx 
            on cargo_ship_position (uid);""")
loc_cargo_conn.commit()
c.execute("""CREATE INDEX ship_position_geom_idx 
            ON cargo_ship_position USING GIST (geog);""")
loc_cargo_conn.commit()
loc_cargo_conn.close()

#%%
print(glob.glob((folder + '/**/*.csv'), recursive=True))