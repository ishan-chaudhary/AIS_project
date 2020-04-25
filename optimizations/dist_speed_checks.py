#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:16:46 2020

@author: patrickmaus
"""
import pandas as pd
import numpy as np
import os

#time tracking
import datetime

# db admin
import psycopg2
from sqlalchemy import create_engine

from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances


def create_sql_alch_engine(database):
    user = 'patrickmaus'
    host = 'localhost'
    port = '5432'
    return create_engine('postgresql://{}@{}:{}/{}'.format(user, host, 
                                                           port, database))
loc_engine = create_sql_alch_engine('ais_test')

#%%

"""This function finds the center of a cluster from dbscan results,
determines the nearest port, and finds the average distance for each
cluster point from its cluster center.  Returns a df."""
tick = datetime.datetime.now()


table = 'dbscan_results_15_2000'
df_results = pd.read_sql(table, loc_engine, columns=['id', 'lat','lon','clust_id'])
#%%
#get all the ports from the world port index
ports = pd.read_sql('wpi', loc_engine, columns=['index_no', 'port_name',
                                                      'latitude','longitude'])
ports = ports.rename(columns={'latitude':'lat','longitude':'lon',
                              'index_no':'port_id'})

#%%

# make a new df from the df_results grouped by cluster id 
# with the mean for lat and long
df_centers = (df_results[['clust_id', 'lat','lon']]
           .groupby('clust_id')
           .mean()
           .rename({'lat':'average_lat', 'lon':'average_lon'}, axis=1)
           .reset_index())    

#%% 

# Now we are going to use sklearn's KDTree to find the nearest neighbor of
# each center for the nearest port.
points_of_int = np.radians(df_centers.loc[:,['average_lat','average_lon']].values)
candidates = np.radians(ports.loc[:,['lat','lon']].values)

# Create the BallTree using the candidates.
tree = BallTree(candidates, metric='haversine')
# this list will store the nearest point of interest to the candidate
nearest_list = []
for i in range(len((points_of_int))):
    # feed the tree one point of int at a time, and get the dist and ind of the
    # candidate.  k=1 indicates return the 1 nearest candidate.  For more, rework
    # the indexing below
    dist, ind = tree.query(points_of_int[i,:].reshape(1, -1), k=1)
    # build a dict with the nearest port id and distance.  add it to a list.
    nearest_dict ={'nearest_port_id':ports.iloc[ind[0][0]].loc['port_id'], 
                   'nearest_port_dist':dist[0][0]*6371.0088}
    nearest_list.append(nearest_dict)
df_nearest = pd.DataFrame(nearest_list)

# merge the nearest ports back to the centers
df_centers = pd.merge(df_centers, df_nearest, how='left', 
                      left_index=True, right_index=True)



#%%
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
                  'max_dist_from_center': haver_result.max(),
                  'average_dist_from_center':np.mean(haver_result)}
    haver_list.append(haver_dict)
    
haver_df = pd.DataFrame(haver_list)

df_centers = pd.merge(df_centers, haver_df, how='left', on='clust_id')
#%%
# each center for the nearest port.
points_of_int = np.radians(df_centers.loc[:,['average_lat','average_lon']].values)
candidates = np.radians(ports.loc[:,['lat','lon']].values)

# Create the BallTree using the candidates.
tree = BallTree(candidates, metric='haversine')
# this list will store the nearest point of interest to the candidate
nearest_list = []
for i in range(len((points_of_int))):
    # feed the tree one point of int at a time, and get the dist and ind of the
    # candidate.  k=1 indicates return the 1 nearest candidate.  For more, rework
    # the indexing below
    dist, ind = tree.query(points_of_int[i,:].reshape(1, -1), k=len(candidates))
    # build a dict with the nearest port id and distance.  add it to a list.
    nearest_dict ={'nearest_port_id':ports.iloc[ind[0][0]].loc['port_id'], 
                   'nearest_port_dist':dist[0][0]*6371.0088}
    nearest_list.append(nearest_dict)
df_nearest = pd.DataFrame(nearest_list)

# merge the nearest ports back to the centers
df_centers = pd.merge(df_centers, df_nearest, how='left', 
                      left_index=True, right_index=True)

dist_km = dist[0]*6371

#%%
bsas = [-34.83333, -58.5166646]
paris = [49.0083899664, 2.53844117956]
bsas_in_radians = np.radians(bsas)
paris_in_radians = np.radians(paris)
result = haversine_distances(bsas_in_radians, paris_in_radians)
result_km = result * 6371000/1000  # multiply by Earth radius to get kilometers