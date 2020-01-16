#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:14:56 2020

@author: patrickmaus
"""

import psycopg2

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
    
#%% Create Port Activity table for sample
port_activity_sample_sql = """
CREATE TABLE port_activity_sample AS
-- We need all of the original fields from ship_position as well 
-- as the port name and port id.  
		SELECT pos.mmsi, pos.time, pa.port_name, pa.port_id, pos.geog
		FROM 
		ship_position_sample as pos 
		LEFT JOIN
-- this query returns all ship positions within 2000 m of a port.
-- duplicates are still possible here.
			(SELECT s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id
			FROM ship_position_sample AS s
			JOIN wpi 
			ON ST_DWithin(s.geog, wpi.geog, 2000))
		as pa
-- we then joint the port activity (pa) with all of the ship positions
-- where the mmsi and time are equal
		ON (pa.mmsi = pos.mmsi) AND
		(pa.time = pos.time) 
		ORDER BY (pos.mmsi, pos.time);
"""

execute_sql(port_activity_sample_sql)






    