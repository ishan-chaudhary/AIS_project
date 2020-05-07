#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:14:56 2020

@author: patrickmaus
"""

import psycopg2
from sqlalchemy import create_engine
import datetime

import aws_credentials as a_c


#%% Make and test conn and cursor
database='ais_test'
loc_conn = psycopg2.connect(host="localhost",database='ais_test')
c = loc_conn.cursor()
if c:
    print('Connection to {} is good.'.format(database))
else:
    print('Error connecting.')
c.close()
#%%

user = a_c.user
host = a_c.host
port = '5432'
database = 'aws_ais_clustering'
password = a_c.password

aws_conn = psycopg2.connect(host=host,database=database, 
                        user=user,password=password)

aws_c = aws_conn.cursor()
if aws_c:
    print('Connection to AWS is good.'.format(database))
else: print('Connection failed.')
aws_c.close()
#%%
def create_aws_engine(database):
    import aws_credentials as a_c
    user = a_c.user
    host = a_c.host
    port = '5432'
    password = a_c.password
    return create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, host, port, database))

aws_engine = create_aws_engine('aws_ais_clustering')

    
#%% Create Port Activity table 
def create_port_activity_table(source_table, destination_table, dist, conn):
    
    port_activity_sample_sql = """
    -- This SQL query has two with selects and then a final select to create the new table.
    -- First create the table.  Syntax requires its creation before any with clauses.
    CREATE TABLE {1} AS
    -- First with clause gets all positions within x meters of any port.  Note there are dupes.
    WITH port_activity as (
    		SELECT s.id, s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id,
    		(ST_Distance(s.geog, wpi.geog)) as dist_meters
    		FROM {0} AS s
    		JOIN wpi 
    		ON ST_DWithin(s.geog, wpi.geog, {2})
    -- Second with clause has a nested select that returns the closest port and groups by mmsi and time.
    -- This result is then joined back to the original data.
    ),  port_activity_reduced as (
    		SELECT pa.id, pa.mmsi, pa.time, pa.port_name, pa.port_id, t_agg.dist_meters FROM
    		(SELECT mmsi, time, min(dist_meters) as dist_meters 
    		FROM port_activity as pa
    		GROUP BY (mmsi, time)) as t_agg, 
    		port_activity as pa
    		WHERE pa.mmsi = t_agg.mmsi AND
    		pa.time = t_agg.time AND
    		pa.dist_meters = t_agg.dist_meters
    )
    -- We need all of the original fields from ship_position as well so this block joins the
    -- results back to ALL positions, regardles if they were near a port.  
    		SELECT pos.id, pos.mmsi, pos.time, pos.geog, pa.port_name, pa.port_id
    		FROM 
    		{0} as pos
    		LEFT JOIN
    		port_activity_reduced as pa
    		ON (pa.mmsi = pos.mmsi) AND
    		(pa.time = pos.time) 
    		ORDER BY (pos.mmsi, pos.time);""".format(source_table, destination_table, dist)
          
    first_tick = datetime.datetime.now()
    print('Starting Processing at: ', first_tick.time())
    
    c = conn.cursor()
    c.execute(port_activity_sample_sql)
    conn.commit()
    c.close()
    
    last_tock = datetime.datetime.now()
    lapse = last_tock - first_tick
    print('Processing Done.  Total time elapsed: ', lapse)
#%% Run query

create_port_activity_table('cargo_tanker_position', 'cargo_port_activity_2k', 2000, loc_conn)
#%%
#create_port_activity_table('ship_position', 'port_activity_5k', 5000, loc_conn)

