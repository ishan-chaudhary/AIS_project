#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:14:56 2020

@author: patrickmaus
"""

import psycopg2
import datetime

#choose db.  this is used for connections throughout the script
database = 'ais_test'

#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database=database)
c = conn.cursor()
if c:
    print('Connection to {} is good.'.format(database))
else:
    print('Error connecting.')
c.close()

#%% Function for executing SQL
def execute_sql(SQL_string):
    c = conn.cursor()
    c.execute(SQL_string)
    conn.commit()
    c.close()
    
#%% Create Port Activity table 
destination_table = 'port_activity_full'
source_table = 'ship_position'    

port_activity_sample_sql = """
-- This SQL query has two with selects and then a final select to create the new table.
-- First create the table.  Syntax requires its creation before any with clauses.
CREATE TABLE {} AS
-- First with clause gets all positions within x meters of any port.  Note there are dupes.
WITH port_activity as (
		SELECT s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id,
		(ST_Distance(s.geog, wpi.geog)) as dist_meters
		FROM ship_position_sample AS s
		JOIN wpi 
		ON ST_DWithin(s.geog, wpi.geog, 2000)
-- Second with clause has a nested select that returns the closest port and groups by mmsi and time.
-- This result is then joined back to the original data.
),  port_activity_reduced as (
		SELECT pa.mmsi, pa.time, pa.port_name, pa.port_id, t_agg.dist_meters FROM
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
		SELECT pos.mmsi, pos.time, pos.geog, pa.port_name, pa.port_id
		FROM 
		{} as pos
		LEFT JOIN
		port_activity_reduced as pa
		ON (pa.mmsi = pos.mmsi) AND
		(pa.time = pos.time) 
		ORDER BY (pos.mmsi, pos.time);""".format(destination_table, source_table)
#%% Run query
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())
execute_sql(port_activity_sample_sql)
last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)





    