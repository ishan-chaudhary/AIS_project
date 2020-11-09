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
import db_config

#%%
def sklearn_dbscan(source_table, new_table_name, eps, min_samples, 
                   uid_list, conn, engine, schema_name, 
                   lat='lat', lon='lon'):
    for uid in uid_list: 
        # next get the data for the uid
        read_sql = """SELECT id, uid, {0}, {1}
                    FROM {2}
                    WHERE uid = '{3}'
                    ORDER by time""".format(lat, lon, source_table, uid[0])
        df = pd.read_sql_query(read_sql, con=engine)
        
        # format data for dbscan
        X = (np.radians(df.loc[:,['lon','lat']].values))
        x_id = df.loc[:,'id'].values
        
        # execute sklearn's clustering_analysis
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', 
                        metric='haversine', n_jobs=-1)
        dbscan.fit(X)
        
        # gather the output as a dataframe
        results_dict = {'id':x_id, 'lat':np.degrees(X[:,1]),
                        'lon':np.degrees(X[:, 0]),'clust_id': dbscan.labels_}      
        df_clusts = pd.DataFrame(results_dict)
        # drop all -1 clust_id, which are all points not in clusters
        df_clusts = df_clusts[df_clusts['clust_id'] != -1]
        df_clusts['uid'] = uid[0]
        
        # write df to databse
        df_clusts.to_sql(name=new_table_name, con=engine, schema=schema_name,
                  if_exists='append', method='multi', index=False )
        
        print('clustering_analysis complete for uid {}.'.format(uid[0]))
        
def sklearn_dbscan_rollup(source_table, new_table_name, eps, min_samples, 
                          conn, engine, schema_name, 
                          lat='average_lat', lon='average_lon'):

    read_sql = """SELECT clust_id, {0}, {1}
                FROM {2}.{3}
                ORDER by clust_id""".format(lat, lon, schema_name, source_table)
    df = pd.read_sql_query(read_sql, con=engine)
    
    # format data for dbscan.  note lon/lat order
    X = (np.radians(df.loc[:,[lon,lat]].values))
    x_id = df.loc[:,'clust_id'].values
    
    # execute sklearn's clustering_analysis
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', 
                    metric='haversine', n_jobs=-1)
    dbscan.fit(X)
    
    # gather the output as a dataframe
    results_dict = {'id':x_id, 'lat':np.degrees(X[:,1]),
                    'lon':np.degrees(X[:, 0]),'super_clust_id': dbscan.labels_}      
    df_clusts = pd.DataFrame(results_dict)
    # drop all -1 clust_id, which are all points not in clusters
    #df_clusts = df_clusts[df_clusts['super_clust_id'] != -1]

    # write df to databse
    df_clusts.to_sql(name=new_table_name, con=engine, schema=schema_name,
              if_exists='replace', method='multi', index=False )
    
    print('clustering_analysis complete for {}.'.format(source_table))

def postgres_dbscan(source_table, new_table_name, eps, min_samples, 
                    uid_list, conn, schema_name, 
                    lat='lat', lon='lon'):
    
    # drop table if it exists
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS {}.{}""".format(schema_name, new_table_name))
    conn.commit()
    c.close()
    
    # create the new table in the new schema to hold the results for each 
    # point in the source table, which will include the cluster id.
    # Postgres or sklearn, we will only write points that have a valid clust_id
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS {}.{}
    (
        id integer,
        uid text,
        lat numeric,
        lon numeric,
        clust_id int
    );""".format(schema_name, new_table_name))
    conn.commit()
              
    # iterate through each uid and insert into the new schema and table
    # the id, uid, lat, lon, and cluster id using the epsilon in degrees.
    # Only write back when a position's cluster id is not null.
    for uid in uid_list:     
        
        dbscan_postgres_sql = """INSERT INTO {7}.{0} (id, uid, lat, lon, clust_id)
        WITH dbscan as (
        SELECT id, uid, {1}, {2},
        ST_ClusterDBSCAN(geom, eps := {3}, minpoints := {4})
        over () as clust_id
        FROM {5}
        WHERE uid = '{6}')
        SELECT * from dbscan 
        WHERE clust_id IS NOT NULL;""".format(new_table_name, lat, lon, str(eps), 
        str(min_samples), source_table, uid[0], schema_name)
        
        # execute dbscan script
        c = conn.cursor()
        c.execute(dbscan_postgres_sql)
        conn.commit()
        c.close()
        print('uid {} complete.'.format(uid[0]))
        
    print('clustering_analysis complete, {} created'.format(new_table_name))

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
    
def get_uid_list(source_table, conn):
    c = conn.cursor()
    c.execute("""SELECT DISTINCT(uid) FROM {};""".format(source_table))
    uid_list = c.fetchall()
    print('{} total uids returned from {}'.format(str(len(uid_list)), source_table))
    c.close()
    return uid_list

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
        print('Old version of schema {} deleted if exists'.format(schema_name))
        
    # make a new schema to hold the results    
    c = conn.cursor()
    c.execute("""CREATE SCHEMA IF NOT EXISTS {};""".format(schema_name))
    conn.commit()
    print ('New schema {} created.'.format(schema_name))
    return schema_name


def execute_dbscan(source_table, eps_samples_params, conn, engine, method='sklearn', 
                   drop_schema=False):    
    # check to make sure the method type is correct
    method_types = ['sklearn_uid','postgres_uid', 'sklearn_rollup']
    if method not in method_types:
        print("Argument 'method' must be 'sklearn_uid', 'sklearn_rollup', or 'postgres_uid'.")
        return
    
    print('{} clustering_analysis begun at:'.format(method), datetime.datetime.now())
    outer_tick = datetime.datetime.now()
    
    # make the new schema for todays date and the method.
    schema_name = create_schema(method, conn, drop_schema=drop_schema, with_date=True)
    
    if method in ['sklearn_uid','postgres_uid']:
        # get the uid list from the source table.
        uid_list = get_uid_list(source_table, conn)
    else: pass

    # itearate through the epsilons and samples given
    for p in eps_samples_params:
        eps_km, min_samples = p
        inner_tick = datetime.datetime.now()
    
        print("""Starting processing on {} clustering_analysis with 
        eps_km={} and min_samples={} """.format(method, str(eps_km), str(min_samples)))
        
        # this formulation will yield epsilon based on km desired. 
        # clustering_analysis in postgres only works with geom, so distance is based on
        # cartesian plan distance calculations.  This is only approximate
        # because the length of degrees are different for different latitudes. 
        # however it should be fine for small distances.
        kms_per_radian = 6371.0088
        eps = eps_km / kms_per_radian
       
        # make the new table name
        new_table_name = (method + str(eps_km).replace('.','_') +
                          '_' + str(min_samples))

        if method == 'postgres':
            postgres_dbscan(source_table, new_table_name, eps, min_samples, 
                            uid_list, conn, schema_name)
        elif method == 'sklearn':
            sklearn_dbscan(source_table, new_table_name, eps, min_samples, 
                           uid_list, conn, engine, schema_name)
        elif method == 'sklearn_rollup':
            sklearn_dbscan_rollup(source_table, new_table_name, eps, min_samples, 
                          conn, engine, schema_name)

        # add geom colum to the new tables
        make_tables_geom(new_table_name, schema_name, conn)
         
        #timekeeping for each iteration
        tock = datetime.datetime.now()
        lapse = tock - inner_tick
        print ('Time elapsed for this iteration: {}'.format(lapse))
        
    #timekeeping for entire approach
    tock = datetime.datetime.now()
    lapse = tock - outer_tick
    print ('Time elapsed for entire process: {}'.format(lapse))


#%%
source_table = 'ship_position_sample'
epsilons_km = [.25, .5, 1, 2, 3, 5, 7]
samples = [2, 5, 10, 25, 50, 100, 250, 500]

eps_samples_params = []
for eps_km in epsilons_km:
    for min_samples in samples: 
        eps_samples_params.append([eps_km, min_samples])
        
#%%
source_table = 'cargo_ship_position'
eps_samples_params = [[2, 100],
                      [.5, 50],
                      [.25,100],
                      [.5, 100]]

#%% execute run against the full data set with parameters the performed the best
# against the sample data.
conn = gsta.connect_psycopg2(db_config.loc_cargo_params)
loc_engine = gsta.connect_engine(db_config.loc_cargo_params)

execute_dbscan(source_table, eps_samples_params, conn, loc_engine, method='sklearn', 
                   drop_schema=True, drop_table=True)
conn.close()

#%%
conn = gsta.connect_psycopg2(db_config.loc_cargo_params)
loc_engine = gsta.connect_engine(db_config.loc_cargo_params)
#%%
schema_name='sklearn_dbscan_results_full_2020_05_12'
eps_km = .5

kms_per_radian = 6371.0088
eps = eps_km / kms_per_radian

sklearn_dbscan_rollup('rollup_0_5_100', eps=eps, min_samples=3, conn=conn, engine=loc_engine, 
                      schema_name=schema_name)

make_tables_geom('super_cluster_0_5_3', schema_name, conn)
conn.close()

#%%

execute_dbscan('sklearn_dbscan_results_full_2020_05_13.rollup_0_5_100', 
               [[2, 2]], conn, loc_engine, method='sklearn', 
               drop_schema=False, drop_table=False, lat='average_lat', lon='average_lon')