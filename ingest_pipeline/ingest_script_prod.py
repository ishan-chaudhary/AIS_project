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

# %%
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)

positions_table = 'cargo_ship_position'
trips_table = 'trips'

# %% drop other tables
# # Keep commented out unless we are re-ingesting everything.
gsta.drop_table('uid_info', conn)
gsta.drop_table(positions_table, conn)
gsta.drop_table('imported_ais', conn)
gsta.drop_table(trips_table, conn)
# %% start processing
current_folder = os.getcwd()
if not os.path.exists(current_folder + '/ingest_script_processing/logs'):
    os.makedirs(current_folder + '/ingest_script_processing/logs')
folder = current_folder + '/ingest_script_processing'

first_tick = datetime.datetime.now()
log_name = folder + '/proc_log_{}.txt'.format(first_tick.strftime("%Y_%m_%d_%H%M"))
log = open(log_name, 'a+')
log.write('Starting processing at: {} \n'.format(first_tick.strftime("%Y_%m_%d_%H%M")))
log.close()
print('Starting processing at: {}'.format(first_tick.strftime("%Y_%m_%d_%H%M")))

# %% create an imported_ais table to hold each file as its read in
c = conn.cursor()
c.execute("""CREATE TABLE if not exists imported_ais (
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
c.execute("""CREATE TABLE IF NOT EXISTS uid_info
(
    uid text,
    ship_name text,
    ship_type text
);""")
conn.commit()

# Create index on uid
c.execute("""CREATE INDEX if not exists uid_info_uid_idx on uid_info (uid);""")
conn.commit()
c.close()

# %% Create "ship_position" table in the  database.
c = conn.cursor()
c.execute(f"""CREATE TABLE IF NOT EXISTS {positions_table}
(   id serial,
    uid text,
    time timestamp,
    geom geometry,
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
    c.execute("""CREATE INDEX if not exists wpi_geog_idx
              ON wpi
              USING GIST (geog);""")
    conn.commit()
    c.execute("""COPY wpi FROM '{}'
        WITH (format csv, header);""".format(wpi_csv_path))
    conn.commit()
    c.close()
    print('WPI created')


# %%
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
    c.execute("""DELETE FROM imported_ais""")
    conn.commit()
    c.execute("""COPY imported_ais FROM '{}'
                WITH (format csv, header);""".format(file_name))
    conn.commit()
    # this will only insert positions from cargo ship types
    c.execute(f"""INSERT INTO {positions_table} (uid, time, geog, lat, lon)
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
completed_files = []
error_files = []

# iterate through each month and each zone.  get the file and then parse them.
# need to add a check that writes each file and counts to a table for tracking iterations
for month in range(1, 13):
    for zone in [9, 10, 11, 14, 15, 16, 17, 18, 19, 20]:

        # clear out any csvs first that are in the directory.
        # since the files unzip to different subfolders, we have to use the blunt glob.glob approach
        # to find the new csv.  Therefore, we are going to nuke any csvs in the directory or any subdirectories
        print('Removing any .csv files in the target folder.')
        for file in (glob.glob((folder + '/**/*.csv'), recursive=True)):
            os.remove(file)
            print('removed file', file[-22:])

        log = open(log_name, 'a+')
        log.write(f'Started month/zone {month}/{zone} at {datetime.datetime.now().strftime("%H%M")} \n')
        log.close()
        try:
            file_name = f'{str(year)}_{str(month).zfill(2)}_{str(zone).zfill(2)}'
            download_path = f'{folder}/{file_name}.zip'
            unzip_path = f'{folder}/{str(year)}'
            url_path = f'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{str(year)}/'
            url_file = f'AIS_{str(year)}_{str(month).zfill(2)}_Zone{str(zone).zfill(2)}'
            link = url_path + url_file + '.zip'
            tick = datetime.datetime.now()
            print(f'Started file {file_name[-22:]} at {tick.strftime("%H%M")}.')

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
                # file_path = folder + '/' + str(year) + '/AIS_ASCII_by_UTM_Month/2017_v2/' + url_file + '.csv'
                # because the file unzips to different named folders, we have to be a bit creative.
                # if there is only one csv file in the root folder, everything is working.
                if len((glob.glob((folder + '/**/*.csv'), recursive=True))) == 1:
                    file = glob.glob((folder + '/**/*.csv'), recursive=True)[0]
                    parse_ais_SQL(file)
                    print(f'Parsing complete for {file_name}.')
                    # delete the csv file
                    os.remove(file)
                    print('CSV file deleted.')
                    log = open(log_name, 'a+')
                    log.write('File {} parsed and removed.\n'.format(file_name[-22:]))
                    log.close()
                else:
                    print('More than one file expected.  Removing all of them.')
                    for file in (glob.glob((folder + '/**/*.csv'), recursive=True)):
                        os.remove(file)
                    print('Too many csv files.  File name added to the error list.')
                    error_files.append(url_file)
                    log = open(log_name, 'a+')
                    log.write('Error for file {}.  Too many csvs.\n'.format(file_name[-22:]))
                    log.close()

                # append the file to the completed file list.
                completed_files.append(url_file)
        except:
            print('Error.  File name added to the error list.')
            error_files.append(url_file)
            log = open(log_name, 'a+')
            log.write(f'Error month/zone {month}/{zone}.')
            log.close()

        # wrap up time keeping and logging
        tock = datetime.datetime.now()
        lapse = tock - tick
        print('Time elapsed: {} \n'.format(lapse))

        log = open(log_name, 'a+')
        log.write('Finished file {} at {} \n'.format(file_name[-22:], datetime.datetime.now().strftime("%H%M")))
        log.write('Time elapsed: {} \n'.format(lapse))
        log.write(' ')
        log.close()

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

# %% additional functions and index building
loc_cargo_conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
gsta.dedupe_table('uid_info', conn=loc_cargo_conn)
make_trips(trips_table, positions_table, conn=loc_cargo_conn)
make_wpi(conn=loc_cargo_conn)
# %%
c = loc_cargo_conn.cursor()
c.execute("""CREATE INDEX if not exists position_uid_idx 
            on {positions_table} (uid);""")
loc_cargo_conn.commit()
c.execute("""CREATE INDEX if not exists position_geom_idx 
            ON {positions_table} USING GIST (geog);""")
loc_cargo_conn.commit()
loc_cargo_conn.close()

# %%
print(glob.glob((folder + '/**/*.csv'), recursive=True))
