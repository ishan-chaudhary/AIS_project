#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:03:35 2020

@author: patrickmaus
"""

#time tracking
import datetime

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

#aws_conn = gsta.connect_psycopg2(gsta_config.aws_ais_cluster_params)
#aws_conn.close()    

#%%
def create_schema(schema_name, conn, drop_schema=False, with_date=True):
    # add the date the run started if desired
    if with_date==True:
        date = str(datetime.date.today()).replace('-','_')
        schema_name = schema_name + '_' + date
        print('Old version of schema {} deleted'.format(schema_name))
    
    # if desired, drop existing schema name
    if drop_schema==True:
        c = conn.cursor()
        c.execute("""DROP SCHEMA IF EXISTS {} CASCADE;""".format(schema_name))
        conn.commit()
    # make a new schema to hold the results    
    c = conn.cursor()
    c.execute("""CREATE SCHEMA IF NOT EXISTS {};""".format(schema_name))
    conn.commit()
    print ('New schema {} created.'.format(schema_name))
    
    return schema_name

#%%
def postgres_dbscan(source_table, new_table_name, eps_km, min_samples, 
                    mmsi_list, conn, schema_name, 
                    lat='lat', lon='lon'):
    # create the new table in the new schema to hold the results for each 
    # point in the source table, which will include the cluster id.
    # NOTE: Any points not in a cluster will have NULL for clust_id.
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS {}.{}
    (
        id integer,
        mmsi text,
        lat numeric,
        lon numeric,
        clust_id int
    );""".format(schema_name, new_table_name))
    conn.commit()

    print("""Starting processing on DBSCAN with eps={} and
          min_samples={} """.format(str(eps_km), str(min_samples)))
          
    # this formulation will yield epsilon based on km desired. 
    # DBSCAN in post gis only works with geom, so distance is based on
    # cartesian plan distance calculations.  This is only approximate
    # because the length of degrees are different for different latitudes. 
    # however it should be fine for small distances.
    kms_per_radian = 6371.0088
    eps = eps_km / kms_per_radian
    
    # iterate through each mmsi and insert into the new schema and table
    # the id, mmsi, lat, lon, and cluster id using the epsilon in degrees.
    # Only write back when a position's cluster id is not null.
    for mmsi in mmsi_list:          
        dbscan_sql = """INSERT INTO {7}.{0} (id, mmsi, lat, lon, clust_id)
        WITH dbscan as (
        SELECT id, mmsi, {1}, {2},
        ST_ClusterDBSCAN(geom, eps := {3}, minpoints := {4})
        over () as clust_id
        FROM {5}
        WHERE mmsi = '{6}')
        SELECT * from dbscan 
        WHERE clust_id IS NOT NULL;""".format(new_table_name, lat, lon, str(eps), 
        str(min_samples), source_table, mmsi[0], schema_name)
        
        # execute dbscan script
        c = conn.cursor()
        c.execute(dbscan_sql)
        conn.commit()
        c.close()
        
        print('MMSI {} complete.'.format(mmsi[0]))
    
    print('DBSCAN complete, {} created'.format(new_table_name))


def make_tables_geom(table, conn):
    # add a geom column to the new table and populate it from the lat and lon columns
    c = conn.cursor()
    c.execute("""ALTER TABLE {}.{} ADD COLUMN
                geom geometry(Point, 4326);""".format(schema_name, table))
    conn.commit()
    c.execute("""UPDATE {}.{} SET
                geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);""".format(schema_name, table))
    conn.commit()
    c.close()


#%%
drop_table = True
source_table = 'ship_position_sample'

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
#conn = gsta.connect_psycopg2(gsta_config.aws_ais_cluster_params)

c = conn.cursor()
c.execute("""SELECT DISTINCT(mmsi) FROM {};""".format(source_table))
mmsi_list = c.fetchall()
print('{} total MMSIs returned from {}'.format(str(len(mmsi_list)), source_table))
c.close()

schema_name = create_schema('dbscan_results', conn, drop_schema=False)

conn.close()
#%%
print('Function run at:', datetime.datetime.now())
epsilons_km = [.25, .5, 1, 2, 3, 5, 7]
samples = [2, 5, 7, 10, 25, 50, 100, 250, 500]

for eps_km in epsilons_km:
    for min_samples in samples:        
        tick = datetime.datetime.now()
    
        # make the new table name
        new_table_name = ('by_mmsi_' + str(eps_km).replace('.','_') +
                          '_' + str(min_samples))
        
        # reestablish conn for each running
        conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
        #conn = gsta.connect_psycopg2(gsta_config.aws_ais_cluster_params)
        
        if drop_table == True:
            # drop table if exists if needed.
            c = conn.cursor()
            c.execute("""DROP TABLE IF EXISTS {}""".format(new_table_name))
            conn.commit()
            c.close()

        postgres_dbscan(source_table, new_table_name, eps_km, min_samples, 
                        mmsi_list, conn, schema_name)
        
        # add geom colum to the new tables
        make_tables_geom(new_table_name, conn)
        
        conn.close()

        #timekeeping
        tock = datetime.datetime.now()
        lapse = tock - tick
        print ('Time elapsed: {}'.format(lapse))

#%% full data
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
c = conn.cursor()
c.execute("""SELECT DISTINCT(mmsi) FROM cargo_ship_position;""")
mmsi_list = c.fetchall()
print('{} total MMSIs returned.'.format(str(len(mmsi_list))))
c.close()
#%%
postgres_dbscan('cargo_ship_position', 'full_run', 3, 10, 
                        mmsi_list, conn, schema_name)
        
