#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 23:03:35 2020

@author: patrickmaus
"""

#time tracking
import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

#%%
def sklearn_dbscan(source_table, new_table_name, eps, min_samples, 
                   mmsi_list, conn, engine, schema_name, 
                   lat='lat', lon='lon'):
    for mmsi in mmsi_list: 
        # next get the data for the mmsi
        read_sql = """SELECT id, mmsi, {0}, {1}
                    FROM {2}
                    WHERE mmsi = '{3}'
                    ORDER by time""".format(lat, lon, source_table, mmsi[0])
        df = pd.read_sql_query(read_sql, con=engine)
        
        # format data for dbscan
        X = (np.radians(df.loc[:,['lon','lat']].values))
        x_id = df.loc[:,'id'].values
        
        # execute sklearn's DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', 
                        metric='haversine')
        dbscan.fit(X)
        
        # gather the output as a dataframe
        results_dict = {'id':x_id, 'lat':np.degrees(X[:,1]),
                        'lon':np.degrees(X[:, 0]),'clust_id': dbscan.labels_}      
        df_results = pd.DataFrame(results_dict)
        # drop all -1 clust_id, which are all points not in clusters
        df_results = df_results[df_results['clust_id'] != -1]
        df_results['mmsi'] = mmsi[0]
        
        # write df to databse
        df_results.to_sql(name=new_table_name, con=engine, schema=schema_name,
                  if_exists='append', method='multi', index=False )
        
        print('DBSCAN complete for MMSI {}.'.format(mmsi[0]))


def postgres_dbscan(source_table, new_table_name, eps, min_samples, 
                    mmsi_list, conn, schema_name, 
                    lat='lat', lon='lon'):
              
    # iterate through each mmsi and insert into the new schema and table
    # the id, mmsi, lat, lon, and cluster id using the epsilon in degrees.
    # Only write back when a position's cluster id is not null.
    for mmsi in mmsi_list:     
        
        dbscan_postgres_sql = """INSERT INTO {7}.{0} (id, mmsi, lat, lon, clust_id)
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
        c.execute(dbscan_postgres_sql)
        conn.commit()
        c.close()
        print('MMSI {} complete.'.format(mmsi[0]))
        
    print('DBSCAN complete, {} created'.format(new_table_name))

def make_tables_geom(table, schema_name, conn):
    # add a geom column to the new table and populate it from the lat and lon columns
    c = conn.cursor()
    c.execute("""ALTER TABLE {}.{} ADD COLUMN
                geom geometry(Point, 4326);""".format(schema_name, table))
    conn.commit()
    c.execute("""UPDATE {}.{} SET
                geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);""".format(schema_name, table))
    conn.commit()
    c.close()
    
def get_mmsi_list(source_table, conn):
    c = conn.cursor()
    c.execute("""SELECT DISTINCT(mmsi) FROM {};""".format(source_table))
    mmsi_list = c.fetchall()
    print('{} total MMSIs returned from {}'.format(str(len(mmsi_list)), source_table))
    c.close()
    return mmsi_list

def create_schema(schema_name, conn, drop_schema=True, with_date=True):
    # add the date the run started if desired
    if with_date==True:
        date = str(datetime.date.today()).replace('-','_')
        schema_name = schema_name + '_' + date
        
    # if desired, drop existing schema name
    if drop_schema==True:
        c = conn.cursor()
        c.execute("""DROP SCHEMA IF EXISTS {} CASCADE;""".format(schema_name))
        conn.commit()
        print('Old version of schema {} deleted'.format(schema_name))
    # make a new schema to hold the results    
    c = conn.cursor()
    c.execute("""CREATE SCHEMA IF NOT EXISTS {};""".format(schema_name))
    conn.commit()
    print ('New schema {} created.'.format(schema_name))
    return schema_name


def execute_dbscan(source_table, epsilons_km, samples, conn, engine, method='sklearn', 
                   drop_schema=False, drop_table=True):    
    # check to make sure the method type is correct
    method_types = ['sklearn','postgres']
    if method not in method_types:
        print("Argument 'method' must be 'sklearn' or 'postgres'.")
        return
    
    print('{} DBSCAN begun at:'.format(method), datetime.datetime.now())
    
    # make the new schema for todays date and the method.
    schema_name = create_schema('{}_dbscan_results'.format(method), conn, 
                                drop_schema=drop_schema)
    # get the mmsi list from the source table.
    mmsi_list = get_mmsi_list(source_table, conn)

    # itearate through the epsilons and samples given
    for eps_km in epsilons_km:
        for min_samples in samples:        
            tick = datetime.datetime.now()
        
            print("""Starting processing on {} DBSCAN with 
            eps_km={} and min_samples={} """.format(method, str(eps_km), str(min_samples)))
            
            # this formulation will yield epsilon based on km desired. 
            # DBSCAN in postgres only works with geom, so distance is based on
            # cartesian plan distance calculations.  This is only approximate
            # because the length of degrees are different for different latitudes. 
            # however it should be fine for small distances.
            kms_per_radian = 6371.0088
            eps = eps_km / kms_per_radian
           
            # make the new table name
            new_table_name = ('by_mmsi_' + str(eps_km).replace('.','_') +
                              '_' + str(min_samples))
            
            if drop_table == True:
                # drop table if exists if needed.
                c = conn.cursor()
                c.execute("""DROP TABLE IF EXISTS {}""".format(new_table_name))
                conn.commit()
                c.close()
            
            # create the new table in the new schema to hold the results for each 
            # point in the source table, which will include the cluster id.
            # Postgres or sklearn, we will only write points that have a valid clust_id
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
            
            if method == 'postgres':
                postgres_dbscan(source_table, new_table_name, eps, min_samples, 
                                mmsi_list, conn, schema_name)
            elif method == 'sklearn':
                sklearn_dbscan(source_table, new_table_name, eps, min_samples, 
                               mmsi_list, conn, engine, schema_name)
    
            # add geom colum to the new tables
            make_tables_geom(new_table_name, schema_name, conn)
             
            #timekeeping for each iteration
            tock = datetime.datetime.now()
            lapse = tock - tick
            print ('Time elapsed for this iteration: {}'.format(lapse))
            
    #timekeeping for entire approach
    tock = datetime.datetime.now()
    lapse = tock - tick
    print ('Time elapsed for entire process: {}'.format(lapse))


#%%
source_table = 'ship_position_sample'
epsilons_km = [.25, .5, 1, 2, 3, 5, 7]
samples = [2, 5, 7, 10, 25, 50, 100, 250, 500]

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)

execute_dbscan(source_table, epsilons_km, samples, conn, loc_engine, method='postgres', 
                   drop_schema=True, drop_table=True)
conn.close()


