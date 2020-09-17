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
import pandas as pd

# folder management
import glob

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

# %%
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.colone_cargo_params)

# %%
# Create enumerated type for the strings in the status column
# c = conn.cursor()
# c.execute("""DROP TYPE IF EXISTS ship_status;""")
# conn.commit()
# c.execute("""CREATE TYPE ship_status AS ENUM (
# 'at anchor',
# 'reserved for future use (10)',
# 'AIS-SART (active); MOB-AIS; EPIRB-AIS',
# 'reserved for future use (13)',
# 'reserved for future use (9)',
# 'aground',
# 'undefined',
# 'under way sailing',
# 'power-driven vessel towing astern',
# 'power-driven vessel pushing ahead or towing alongside',
# 'under way using engine',
# 'restricted maneuverability',
# 'engaged in fishing',
# 'constrained by her draught',
# 'not under command',
# 'moored');""")
# conn.commit()

# Create "imported_data" table in the  database.
c.execute("""CREATE TABLE IF NOT EXISTS imported_data
(uid text, 
time timestamp, 
lat numeric, 
lon numeric,
sog numeric,
cog numeric, 
status ship_status,
anchored boolean,
moored boolean,
underway boolean);""")
conn.commit()
c.close()
# %% drop other tables
# Keep commented out unless we are re-ingesting everything.
# gsta.drop_table('uid_info', conn)
# gsta.drop_table('imported_data', conn)
# gsta.drop_table('uid_positions_status', conn)
# gsta.drop_table('uid_trips', conn)
# %% set up proc and error log
current_folder = '/home/patrickfmaus/AIS_project'
if not os.path.exists(current_folder + '/script_processing/logs'):
    os.makedirs(current_folder + '/script_processing/logs')

# %%
ingest_folder = current_folder + '/script_processing'
log_folder = current_folder + '/script_processing/logs'

first_tick = datetime.datetime.now()
# proc log
log_name = log_folder + '/proc_log_{}.txt'.format(first_tick.strftime("%Y_%m_%d_%H%M"))
log = open(log_name, 'a+')
log.write('Starting overall processing at: {} \n'.format(first_tick.strftime("%Y_%m_%d_%H%M")))
log.close()
print('Starting processing at: {}'.format(first_tick.strftime("%Y_%m_%d_%H%M")))
# error log
error_log = log_folder + '/error_log_{}.txt'.format(first_tick.strftime("%Y_%m_%d_%H%M"))
log = open(error_log, 'a+')
log.write('Starting overall processing at: {} \n'.format(first_tick.strftime("%Y_%m_%d_%H%M")))
log.close()

# Create "uid_position" table in the  database.
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS uid_positions_status
(id serial primary key, 
uid text, 
time timestamp, 
geom geometry, 
lat numeric, 
lon numeric,
sog numeric,
cog numeric,
status ship_status,
anchored boolean,
moored boolean,
underway boolean);""")
conn.commit()
c.close()

# %% create WPI table function
wpi_csv_path = current_folder + '/WPI_data/wpi_clean.csv'


def make_sites(conn, file=wpi_csv_path):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS sites (
                    site_id 	int primary key,
                    region_no	int,
                    port_name	text,
                    country		text,
                    latitude	numeric,
                    longitude	numeric,
                    geog		geography,
                    geom		geometry);""")
    c.execute("""CREATE INDEX if not exists sites_geom_idx
              ON sites
              USING GIST (geom);""")
    conn.commit()
    c.execute("""COPY sites FROM '{}'
        WITH (format csv, header);""".format(file))
    conn.commit()
    c.close()
    print('Sites created')


make_sites(conn=conn)


# %%
def download_url(link, download_path, unzip_path, file_name, chunk_size=10485760):
    # chunk size is 10 mb
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
def parse_sql(file_name, conn=conn):
    c = conn.cursor()
    # need to clear out imported data first.
    c.execute("""TRUNCATE imported_data;""")
    conn.commit()
    print('Copying data to temp table...')
    generator = pd.read_csv(file_name, chunksize=500000)
    for df in generator:
        # subset the df to just the columns needed for processing and rename them
        df = df[['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Status', 'VesselType']]
        df.columns = ['uid', 'time', 'lat', 'lon', 'sog', 'cog', 'status', 'ship_type']
        # filter by just cargo ships as ship_type, then drop the column
        df = df[df['ship_type'].isin([70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1003, 1004, 1016])]
        df.drop('ship_type', inplace=True, axis=1)
        # convert time to pandas datetime
        df['time'] = pd.to_datetime(df['time'])
        # set columns for anchored, moored, and underway, and set values to True when present
        df.loc[df['status'] == 'at anchor', 'anchored'] = True
        df.loc[df['status'] == 'moored', 'moored'] = True
        df.loc[df['status'] == 'under way sailing', 'underway'] = True
        df.loc[df['status'] == 'under way using engine', 'underway'] = True
        df.loc[df['status'] == 'restricted maneuverability', 'underway'] = True
        df.loc[df['status'] == 'constrained by her draught', 'underway'] = True
        # write df back to the database
        df.to_sql(name='imported_data', con=loc_engine, if_exists='append', method='multi', index=False)
    print('Copying complete!')
    # this will only insert positions from cargo ship types
    c.execute("""INSERT INTO uid_positions_status (uid, time, geom, lat, lon, sog, cog, status, anchored, moored, underway)
                SELECT uid, 
                time, 
                ST_SetSRID(ST_MakePoint(lon, lat), 4326), 
                lat, 
                lon ,
                sog,
                cog,
                status,
                anchored,
                moored,
                underway
                FROM imported_data;""")
    conn.commit()
    c.close()


# %% Populate ship_trips table from ship position table
def make_trips(new_table_name, source_table, conn):
    c = conn.cursor()
    c.execute(f"""DROP TABLE IF EXISTS {new_table_name};""")
    conn.commit()
    c.execute(f"""CREATE TABLE {new_table_name} AS
    SELECT 
        uid,
        position_count,
        ST_Length(ST_Transform(line, 4326))/1000 AS line_length_km,
        first_date,
        last_date,
        last_date - first_date as time_diff,
        line
        FROM (
                SELECT pos.uid as uid,
                COUNT (pos.geom) as position_count,
                ST_MakeLine(pos.geom ORDER BY pos.time) AS line,
                MIN (pos.time) as first_date,
                MAX (pos.time) as last_date
                FROM {source_table} as pos
                GROUP BY pos.uid) AS foo;""")
    conn.commit()
    c.execute(f"""CREATE INDEX if not exists trips_uid_idx on {new_table_name} (uid);""")
    conn.commit()
    c.close()


# %% run all the functions using the function tracker
# set variables for functions
year = 2017

# iterate through each month and each zone.  get the file and then parse them.
# need to add a check that writes each file and counts to a table for tracking iterations
for month in range(1, 13):
    for zone in [9, 10, 11, 14, 15, 16, 17, 18, 19, 20]:
        tick = datetime.datetime.now()
        # clear out any csvs first that are in the directory.
        # since the files unzip to different subfolders, we have to use the blunt glob.glob approach
        # to find the new csv.  Therefore, we are going to nuke any csvs in the directory or any subdirectories
        print('Removing any .csv files in the target folder.')
        for file in (glob.glob((ingest_folder + '/**/*.csv'), recursive=True)):
            os.remove(file)
            print('removed file', file[-22:])
        log = open(log_name, 'a+')
        log.write(f'Started month/zone {month}/{zone} at {datetime.datetime.now().strftime("%H%M")} \n')
        log.close()
        # set the variable for this round of iteration
        file_name = f'{str(year)}_{str(month).zfill(2)}_{str(zone).zfill(2)}'
        download_path = f'{ingest_folder}/{file_name}.zip'
        unzip_path = f'{ingest_folder}/{str(year)}'
        url_path = f'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{str(year)}/'
        url_file = f'AIS_{str(year)}_{str(month).zfill(2)}_Zone{str(zone).zfill(2)}'
        link = url_path + url_file + '.zip'

        try:
            print(f'Started file {file_name[-22:]} at {tick.strftime("%H%M")}.')
            # use the download_url function to download the zip file, unzip it, and
            # delete the zip file
            download_url(link, download_path, unzip_path, file_name)
            print('Starting to parse raw data...')
            # because the file unzips to different named folders, we have to be a bit creative.
            # if there is only one csv file in the root folder, everything is working.
            if len((glob.glob((ingest_folder + '/**/*.csv'), recursive=True))) == 1:
                file = glob.glob((ingest_folder + '/**/*.csv'), recursive=True)[0]
                parse_sql(file)
                print(f'Parsing complete for {file_name}.')
                # delete the csv file
                os.remove(file)
                print('CSV file deleted.')
                log = open(log_name, 'a+')
                log.write('File {} parsed and removed.\n'.format(file_name[-22:]))
                log.close()
            else:
                print('More than one file expected.  Removing all of them.')
                for file in (glob.glob((ingest_folder + '/**/*.csv'), recursive=True)):
                    os.remove(file)
                print('Too many csv files.  File name added to the error list.')
                log = open(log_name, 'a+')
                log.write('Error for file {}.  Too many csvs.\n'.format(file_name[-22:]))
                log.close()
                log = open(error_log, 'a+')
                log.write('{} \n'.format(file_name[-22:]))
                log.close()
        except Exception:
            print('Error.  Logging in proc and error log.')
            log = open(log_name, 'a+')
            log.write(f'Error month/zone {month}/{zone}. \n')
            log.close()
            log = open(error_log, 'a+')
            log.write('{} \n'.format(file_name[-22:]))
            log.close()
        # wrap up time keeping and logging
        tock = datetime.datetime.now()
        lapse = tock - tick
        print('Time elapsed: {} \n'.format(lapse))
        log = open(log_name, 'a+')
        log.write('Finished file {} at {} \n'.format(file_name[-22:], datetime.datetime.now().strftime("%H%M")))
        log.write('Time elapsed: {} \n'.format(lapse))
        log.write('\n')
        log.close()

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)
conn.close()
# %% additional functions and index building
# this will populate the roll up table
tick = datetime.datetime.now()
print('Making trips table')
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
make_trips('uid_trips', 'uid_positions_status', conn=conn)
print('Trips table built.')
tock = datetime.datetime.now()
lapse = tock - tick
print('Time elapsed: {} \n'.format(lapse))
conn.close()
# %% indices.  These can take a long time to build.
tick = datetime.datetime.now()
print('Adding indices to uid_positions_status')

conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
c = conn.cursor()
c.execute("""CREATE INDEX if not exists position_uid_idx 
            on uid_positions_status (uid);""")
conn.commit()
# c.execute("""CREATE INDEX if not exists position_geom_idx
#             ON uid_positions_status USING GIST (geom);""")
# conn.commit()
print('Indices built.')
tock = datetime.datetime.now()
lapse = tock - tick
print('Time elapsed: {} \n'.format(lapse))
conn.close()
