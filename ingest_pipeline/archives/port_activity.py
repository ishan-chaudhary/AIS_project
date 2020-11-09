#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:14:56 2020

@author: patrickmaus
"""

import datetime

# Geo-Spatial Temporal Analysis package
import gsta
import db_config

aws_conn = gsta.connect_psycopg2(db_config.aws_ais_cluster_params)
loc_conn = gsta.connect_psycopg2(db_config.loc_cargo_params)
aws_conn.close()    
loc_conn.close()
#%% Create Port Activity table 
def create_port_activity_table(source_table, destination_table, dist, conn):
    
    port_activity_sample_sql = """
    -- This SQL query has two with selects and then a final select to create the new table.
    -- First create the table.  Syntax requires its creation before any with clauses.
    CREATE TABLE {1} AS
    -- First with clause gets all positions within x meters of any port.  Note there are dupes.
    WITH port_activity as (
    		SELECT s.id, s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id,
    		(ST_Distance(s.geom::geography, wpi.geog)) as dist_meters
    		FROM {0} AS s
    		JOIN wpi 
    		ON ST_DWithin(s.geom, wpi.geom, {2})
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
    		SELECT pos.id, pos.mmsi, pos.time, pos.geom::geography, pa.port_name, pa.port_id
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
    
    print('Table done...building index...')
    index_sql = 'CREATE INDEX {0}_mmsi_idx on {0} (mmsi);'.format(destination_table)
    c = conn.cursor()
    c.execute(index_sql)
    conn.commit()
    c.close()
    
    last_tock = datetime.datetime.now()
    lapse = last_tock - first_tick
    print('All Processing Done.  Total time elapsed: ', lapse)
#%% Run query
loc_cargo_conn = gsta.connect_psycopg2(db_config.loc_cargo_params)
create_port_activity_table('cargo_ship_position', 'cargo_port_activity_5k', 5000, loc_cargo_conn)
#%%
#create_port_activity_table('ship_position', 'port_activity_5k', 5000, loc_conn)

