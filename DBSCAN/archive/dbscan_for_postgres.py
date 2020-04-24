#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

#time tracking
import datetime

# db admin
import psycopg2
from sqlalchemy import create_engine

from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances

#%% Make and test conn and cursor using psycopg, 
# and create an engine using sql alchemy

database='ais_test'
loc_conn = psycopg2.connect(host="localhost",database=database)
c = loc_conn.cursor()
if c:
    print('Connection to {} is good.'.format(database))
else:
    print('Error connecting.')
c.close()

def create_sql_alch_engine(database):
    user = 'patrickmaus'
    host = 'localhost'
    port = '5432'
    return create_engine('postgresql://{}@{}:{}/{}'.format(user, host, 
                                                           port, database))
loc_engine = create_sql_alch_engine('ais_test')

#%% This function will be used to write results to the database
def df_to_table_with_geom(df, name, eps, min_samples, conn):
    # add the eps and min_samples value to table name
    new_table_name = ('dbscan_results_' + name + '_' + 
                      str(eps).replace('.','_') + '_' + str(min_samples))
    
    # drop table if an old one exists
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS {}""".format(new_table_name))
    conn.commit()
    c.close()
    # make a new table with the df
    df.to_sql(new_table_name, loc_engine)
    # add a geom column to the new table and populate it from the lat and lon columns
    c = conn.cursor()
    c.execute("""ALTER TABLE {} ADD COLUMN 
                geom geometry(Point, 4326);""".format(new_table_name))
    conn.commit()
    c.execute("""UPDATE {} SET 
                geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);""".format(new_table_name))
    conn.commit()
    c.close()
#%%             
def get_dbscan_results(table, engine):
    df_results = pd.read_sql(table, engine, 
                    columns=['id', 'lat','lon','clust_id'])
    return df_results
#%% center and purity calc functions
def center_calc(df_results):
    """This function finds the center of a cluster from dbscan results,
    determines the nearest port, and finds the average distance for each
    cluster point from its cluster center.  Returns a df."""
    
    # make a new df from the df_results grouped by cluster id 
    # with the mean for lat and long
    df_centers = (df_results[['clust_id', 'lat','lon']]
               .groupby('clust_id')
               .mean()
               .rename({'lat':'average_lat', 'lon':'average_lon'}, axis=1)
               .reset_index())     
    
    # Now we are going to use sklearn's KDTree to find the nearest neighbor of
    # each center for the nearest port.
    points_of_int = np.radians(df_centers.loc[:,['average_lat','average_lon']].values)
    candidates = np.radians(ports.loc[:,['lat','lon']].values)
    tree = BallTree(candidates, leaf_size=30, metric='haversine')
    
    nearest_list = []
    for i in range(len((points_of_int))):
        dist, ind = tree.query( points_of_int[i,:].reshape(1, -1), k=1)
        nearest_dict ={'nearest_port_id':ports.iloc[ind[0][0]].loc['port_id'], 
                       'nearest_port_dist':dist[0][0]*6371.0088}
        nearest_list.append(nearest_dict)
    df_nearest = pd.DataFrame(nearest_list)
    
    df_centers = pd.merge(df_centers, df_nearest, how='left', 
                          left_index=True, right_index=True)
    
    # find the average distance from the centerpoint
    # We'll calculate this by finding all of the distances between each point in 
    # df_results and the center of the cluster.  We'll then take the min and the mean.
    haver_list = []
    for i in df_centers['clust_id']:
        X = (np.radians(df_results[df_results['clust_id']==i]
                        .loc[:,['lat','lon']].values))
        Y = (np.radians(df_centers[df_centers['clust_id']==i]
                        .loc[:,['average_lat','average_lon']].values))
        haver_result = (haversine_distances(X,Y)) * 6371.0088 #km to radians
        haver_dict = {'clust_id': i, 'min_dist_from_center': haver_result.min(), 
                      'average_dist_from_center':np.mean(haver_result)}
        haver_list.append(haver_dict)
        
    haver_df = pd.DataFrame(haver_list)
    
    try:
        df_centers = pd.merge(df_centers, haver_df, how='left', on='clust_id')
    except: 
        print('no clusters.  error in center calcs.')
    return df_centers

def purity_calc(df_results):
    """This function takes df_results and calculates how many points are near
    the same port"""
    
    df_purity = pd.merge(df_results[['clust_id','id']], 
                # df_port_activity is our full dataset, and when filtered to port_id>0
                # it returns all positions identifed within a certain dist of a port
                df_port_activity[df_port_activity['port_id'] > 0].loc[:,['id', 'port_name', 'port_id']], 
                         how='left', on='id')
    # we'll fill any nans with -1 for port id and NONE for port_name
    df_purity['port_id'] = df_purity['port_id'].fillna(-1)
    df_purity['port_name'] = df_purity['port_name'].fillna('NONE')
    
    df_purity_grouped = (df_purity
                         .groupby(['clust_id', 'port_id','port_name'])
                         .count()
                         .reset_index()
                         .rename({'id':'counts_at_port'}, axis=1))
    
    clust_size = (df_purity[['id','clust_id']]
                  .groupby('clust_id')
                  .count()
                  .reset_index()
                  .rename({'id':'total_clust_count'}, axis=1))
    
    df_purity_grouped = pd.merge(df_purity_grouped, clust_size, 
                                 how='left', on='clust_id')
    
    df_purity_grouped['proportion_near_top_port'] = (df_purity_grouped['counts_at_port'] 
                                                     / df_purity_grouped['total_clust_count'])
    
    counts_per_port = df_purity_grouped.groupby('clust_id').size()
    df_purity_grouped = pd.merge(df_purity_grouped, counts_per_port.rename('counts_per_port'), 
                                 how='left', on='clust_id')
    
    df_purity_grouped_top = (df_purity_grouped
                             .sort_values('proportion_near_top_port', ascending=False)
                             .drop_duplicates('clust_id')
                             .sort_values('clust_id')
                             .rename({'port_id':'port_id_with_most_points',
                                      'port_name':'port_name_with_most_points'}, axis=1))
    return df_purity_grouped_top

#%% Read in required data
# make a df with  port activity
df_port_activity = pd.read_sql('port_activity_sample_5k', loc_engine, 
                    columns=['id', 'port_name','port_id'])

#get all the ports from the world port index
ports = pd.read_sql('wpi', loc_engine, columns=['index_no', 'port_name',
                                                      'latitude','longitude'])
ports = ports.rename(columns={'latitude':'lat','longitude':'lon',
                              'index_no':'port_id'})

#%% Run this code when we generate our own df_results from dbscan
rollup_list = []
epsilons = [2, 5, 7, 10, 15]
samples = [50, 100, 250, 500, 1000, 2000]
for e in epsilons:
    for s in samples:      
            
        tick = datetime.datetime.now()
        
        table = 'dbscan_results_{}_{}'.format(str(e), str(s))
        df_results = pd.read_sql(table, loc_engine, columns=['id', 'lat','lon','clust_id'])
        
        #determine the cluster center point, and find the distance to nearest port
        print('Starting distance calculations... ')
        df_centers = center_calc(df_results)
        
        #Look at how many points are designated as near ports in the database
        print('Starting purity calculations...')        
        df_purity = purity_calc(df_results)

        # roll up 
        df_rollup = pd.merge(df_purity, df_centers, how='left', on='clust_id')
        df_rollup.to_csv('./rollups/rollup_full_{}_{}.csv'.format(str(e), str(s)))
        
        #timekeeping
        tock = datetime.datetime.now()
        lapse = tock - tick
        print('All processing for this run complete.')
        print ('Time elapsed: {}'.format(lapse))
    
        rollup_dict = {'eps_km':e, 'min_samples':s, 'time':lapse, 
                        'numb_obs':len(df_results), 
                        'average_cluster_count':np.mean(df_rollup['total_clust_count']),
                        'average_ports_per_cluster':np.mean(df_rollup['counts_per_port']),
                        'average_points_per_port':np.mean(df_rollup['counts_at_port']),
                        'average_prop_per_port':np.mean(df_rollup['proportion_near_top_port']),
                        'average_nearest_port_from_center':np.mean(df_rollup['nearest_port_dist']),
                        'average_cluster_density':np.mean(df_rollup['average_dist_from_center']),
                        'numb_clusters':len(np.unique(df_rollup['clust_id'])),
                        'ports_with_most_points':df_rollup['port_name_with_most_points'].to_list()}
        rollup_list.append(rollup_dict)
        
        #df_to_table_with_geom(df_rollup, 'rollup', e, s, loc_conn)
        
        print('Finished with round ', len(rollup_list))
        print('')
#%%
final_df = pd.DataFrame(rollup_list)
final_df.to_csv('./rollups/summary_full_5k.csv')

