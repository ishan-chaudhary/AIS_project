#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

#time tracking
import datetime

from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances

import warnings
warnings.filterwarnings('ignore')

#%% Make and test conn and cursor using psycopg, 
# and create an engine using sql alchemy

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

#aws_conn = gsta.connect_psycopg2(gsta_config.aws_ais_cluster_params)
#loc_conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
#aws_conn.close()    
#loc_conn.close()

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)


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
    
    # # find the average distance from the centerpoint
    # # We'll calculate this by finding all of the distances between each point in 
    # # df_results and the center of the cluster.  We'll then take the min and the mean.
    # haver_list = []
    # for i in df_centers['clust_id']:
    #     X = (np.radians(df_results[df_results['clust_id']==i]
    #                     .loc[:,['lat','lon']].values))
    #     Y = (np.radians(df_centers[df_centers['clust_id']==i]
    #                     .loc[:,['average_lat','average_lon']].values))
    #     haver_result = (haversine_distances(X,Y)) * 6371.0088 #km to radians
    #     haver_dict = {'clust_id': i, 'min_dist_from_center': haver_result.min(), 
    #                   'max_dist_from_center': haver_result.max(),
    #                   'average_dist_from_center':np.mean(haver_result)}
    #     haver_list.append(haver_dict)
    
    # # merge the haver results back to df_centers
    # haver_df = pd.DataFrame(haver_list)
    # df_centers = pd.merge(df_centers, haver_df, how='left', on='clust_id')
    
    # create "total cluster count" column through groupby
    clust_size = (df_results[['id','clust_id']]
              .groupby('clust_id')
              .count()
              .reset_index()
              .rename({'id':'total_clust_count'}, axis=1))
    # merge results back to df_Centers
    df_centers = pd.merge(df_centers, clust_size, how='left', on='clust_id')
    
    return df_centers

def calc_harmonic_mean(precision, recall):
    return 2 *((precision*recall)/(precision+recall))

def calc_stats(df_rollup, ports_5k):
        # determine the recall, precision, and f-measure
        # drop all duplicates in the rollup df to get just the unique port_ids
        # join to the list of all ports within a set distance of positions.
        # the > count allows to filter out noise where only a handful of positions
        # are near a given port.  Increasing this will increase recall because there
        # are fewer "hard" ports to indetify with very little activity.
        df_stats = pd.merge(df_rollup.drop_duplicates('nearest_port_id'), 
                            ports_5k[ports_5k['count'] > 0], 
                            how='outer', on='nearest_port_id', indicator=True)        
        # this df lists where the counts in the merge.
        # left_only are ports only in the dbscan.  (false positives for dbscan)
        # right_only are ports only in the ports near positions.  (false negatives for dbscan)
        # both are ports in both datasets.  (true positives for dbscan)
        values = (df_stats['_merge'].value_counts())
        # recall is the proporation of relevant items selected
        # it is the number of true positives divided by TP + FN
        stats_recall = values['both']/(values['both']+values['right_only'])
        # precision is the proportion of selected items that are relevant.
        # it is the number of true positives our of all items selected by dbscan.
        stats_precision = values['both']/len(df_centers.drop_duplicates('nearest_port_id'))
        # now find the f_measure, which is the harmonic mean of precision and recall
        stats_f_measure = calc_harmonic_mean(stats_precision, stats_recall)
        
        return stats_f_measure, stats_precision, stats_recall

#%% Read in required data
# variable delcaration
schema_name = 'dbscan_results_2020_05_12'
ports_near_positions = 'ports_5k_positions'

#get all the ports from the world port index
ports = pd.read_sql('wpi', loc_engine, columns=['index_no', 'port_name',
                                                'latitude','longitude'])
ports = ports.rename(columns={'latitude':'lat','longitude':'lon',
                              'index_no':'port_id'})

# get the list of ports within 5k of all positions
ports_5k = pd.read_sql_table(table_name=ports_near_positions, con=loc_engine,
                             columns=['port_name', 'nearest_port_id', 'count'])

#%% Run this code when we generate our own df_results from dbscan
rollup_list = []
df_all_results = pd.DataFrame()

path = '/Users/patrickmaus/Documents/projects/AIS_project/DBSCAN/rollups/{}/'.format(str(datetime.datetime.now().date()))
if not os.path.exists(path):
    os.makedirs(path)

epsilons_km = [.25, .5, 1, 2, 3, 5, 7]
samples = [2, 5, 7, 10, 25, 50, 100, 250, 500]

for eps_km in epsilons_km:
    for min_samples in samples:  

        print("""Starting analyzing DBSCAN results with eps_km={} and min_samples={}"""
              .format(str(eps_km), str(min_samples)))
        tick = datetime.datetime.now()
        
        # make table name, and pull the results from the correct sql table.
        table = ('by_mmsi_' + str(eps_km).replace('.','_') + '_' + str(min_samples))
        
    
        df_results = pd.read_sql_table(table_name=table, con=loc_engine, schema=schema_name, 
                                 columns=['id', 'mmsi', 'lat','lon', 'clust_id'])

        # since we created clusters by mmsi, we are going to need to redefine 
        # clust_id to include the mmsi and clust_id        
        df_results['clust_id'] = (df_results['mmsi'] + '_' + 
                                  df_results['clust_id'].astype(int).astype(str))

        #determine the cluster center point, and find the distance to nearest port
        print('Starting distance calculations... ')
        df_rollup = center_calc(df_results)
        print('Finished distance calculations. ')
        
        # calculate dats
        print('Starting stats calculations...')
        stats_f_measure, stats_precision, stats_recall = calc_stats(df_rollup, ports_5k)
        print("finished stats calculations.")
        
        # roll up 
        df_rollup['eps_km'] = eps_km
        df_rollup['min_samples'] = min_samples
        df_rollup.to_csv(path+'rollup_full_{}_{}.csv'.format(str(eps_km), str(min_samples)))
        
        #timekeeping
        tock = datetime.datetime.now()
        lapse = tock - tick
        print('All processing for this run complete.')
        print ('Time elapsed: {}'.format(lapse))
   
        # the rollup dict contains multiple different metrics options.
        rollup_dict = {'eps_km':eps_km, 'min_samples':min_samples, 
                       'params' : (str(eps_km) + '_' + str(min_samples)),
                        # number of clusters in the run
                        'numb_clusters':len(np.unique(df_rollup['clust_id'])),
                        # average positions per each cluster in each run
                        'average_cluster_count':np.mean(df_rollup['total_clust_count']),
                        
                        # distance metrics
                        # closest port.  Closer the better.
                        'average_nearest_port_from_center':np.mean(df_rollup['nearest_port_dist']),
                        # average and max distance from cluster center.
                        #'average_dist_from_center':np.mean(df_rollup['average_dist_from_center']),
                        #'average_max_dist_from_center':np.mean(df_rollup['max_dist_from_center']),
                        
                        # stats
                        'f_measure':stats_f_measure,
                        'precision':stats_precision,
                        'recall':stats_recall
                        }
       
        rollup_list.append(rollup_dict)
               
        print('Finished with round ', len(rollup_list))
        print('')
        

#%% Make the final_df from the rollup and save to csv.
final_df = pd.DataFrame(rollup_list).round(3)
final_df['params'] = (final_df['eps_km'].astype('str') + '_' 
                      + final_df['min_samples'].astype('str'))
final_df.set_index('params', inplace=True)
final_df.to_csv(path+'summary_5k.csv')
#%%
print(final_df.sort_values('f_measure', ascending=False))

#%%
print('Starting stats calculations...')
stats_f_measure, stats_precision, stats_recall = calc_stats(df_rollup, ports_5k)
print("finished stats calculations.")
