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

# %% drop other tables
# Keep commented out unless we are re-ingesting everything.
#gsta.drop_table('uid_info', conn)
#gsta.drop_table('imported_data', conn)
#gsta.drop_table('uid_positions', conn)
#gsta.drop_table('uid_trips', conn)
# %% set up proc and error log
current_folder = os.getcwd()
if not os.path.exists(current_folder + '/ingest_script_processing/logs'):
    os.makedirs(current_folder + '/ingest_script_processing/logs')
folder = current_folder + '/ingest_script_processing/logs'

first_tick = datetime.datetime.now()
# proc log
log_name = folder + '/proc_log_{}.txt'.format(first_tick.strftime("%Y_%m_%d_%H%M"))
log = open(log_name, 'a+')
log.write('Starting overall processing at: {} \n'.format(first_tick.strftime("%Y_%m_%d_%H%M")))
log.close()
print('Starting processing at: {}'.format(first_tick.strftime("%Y_%m_%d_%H%M")))
# error log
error_log = folder + '/error_log_{}.txt'.format(first_tick.strftime("%Y_%m_%d_%H%M"))
log = open(error_log, 'a+')
log.write('Starting overall processing at: {} \n'.format(first_tick.strftime("%Y_%m_%d_%H%M")))
log.close()

# %% create an imported_data table to hold each file as its read in
c = conn.cursor()
c.execute("""CREATE TABLE if not exists imported_data (
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
# Create "ship_info" table in the database.
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

# Create "uid_position" table in the  database.
c = conn.cursor()
c.execute(f"""CREATE TABLE IF NOT EXISTS uid_positions
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
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)


def make_sites(conn, file=wpi_csv_path):
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS sites (
                    site_id 	int,
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
def parse_SQL(file_name, conn=conn):
    c = conn.cursor()
    # need to clear out imported data first.
    c.execute("""TRUNCATE TABLE imported_data;""")
    conn.commit()
    c.execute("""COPY imported_data FROM '{}'
                WITH (format csv, header);""".format(file_name))
    conn.commit()
    # this will only insert positions from cargo ship types
    c.execute(f"""INSERT INTO uid_positions (uid, time, geom, lat, lon)
                SELECT uid, 
                time, 
                ST_SetSRID(ST_MakePoint(lon, lat), 4326), 
                lat, 
                lon 
                FROM imported_data
                where ship_type IN (
                '70','71','72','73','74','75','76','77','78','79',
                '1003','1004','1016');""")
    conn.commit()
    # c.execute("""INSERT INTO uid_info (uid, ship_name, ship_type)
    #             SELECT DISTINCT uid, ship_name, ship_type from uid_positions
    #             where ship_type IN ('70','71','72','73','74','75','76','77',
    #             '78','79','1003','1004','1016');""")
    # conn.commit()
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
for month in range(6, 13):
    for zone in [9, 10, 11, 14, 15, 16, 17, 18, 19, 20]:
        tick = datetime.datetime.now()
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
            print(f'Started file {file_name[-22:]} at {tick.strftime("%H%M")}.')

            # use the download_url function to download the zip file, unzip it, and
            # delete the zip file
            download_url(link, download_path, unzip_path, file_name)
            print('Starting to parse raw data...')

            # because the file unzips to different named folders, we have to be a bit creative.
            # if there is only one csv file in the root folder, everything is working.
            if len((glob.glob((folder + '/**/*.csv'), recursive=True))) == 1:
                file = glob.glob((folder + '/**/*.csv'), recursive=True)[0]
                parse_SQL(file)
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
                log = open(log_name, 'a+')
                log.write('Error for file {}.  Too many csvs.\n'.format(file_name[-22:]))
                log.close()
                log = open(error_log, 'a+')
                log.write('{} \n'.format(file_name[-22:]))
                log.close()

        except:
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
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
#gsta.dedupe_table('uid_info', conn=conn)
make_trips('uid_trips', 'uid_positions', conn=conn)
conn.close()
# %% indices.  These can take a long time to build.
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
c = conn.cursor()
c.execute("""CREATE INDEX if not exists position_uid_idx 
            on uid_positions (uid);""")
conn.commit()
# c.execute("""CREATE INDEX if not exists position_geom_idx
#             ON uid_positions USING GIST (geom);""")
# conn.commit()
conn.close()

#%%
month = 6
zone = 19
year = 2017
file_name = f'{str(year)}_{str(month).zfill(2)}_{str(zone).zfill(2)}'
download_path = f'{folder}/{file_name}.zip'
unzip_path = f'{folder}/{str(year)}'
url_path = f'https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{str(year)}/'
url_file = f'AIS_{str(year)}_{str(month).zfill(2)}_Zone{str(zone).zfill(2)}'
link = url_path + url_file + '.zip'


#%%
tick = datetime.datetime.now()
download_url(link, download_path, unzip_path, file_name, chunk_size=128)
tock = datetime.datetime.now()
lapse = tock - tick
print(lapse)