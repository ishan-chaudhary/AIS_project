#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

#time tracking
import datetime

# db admin
import psycopg2
import sqlalchemy as db

from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN

#%% Find Center of Each Cluster and compare to nearest Port
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # 6371 Radius of earth in kilometers. Use 3956 for miles
    return c * r

def determine_min_distances(df1, name_1, df2, name_2):
    min_distances = []
    for i in range(len(df1)):
        lon1 = df1['lon'].loc[i]
        lat1 = df1['lat'].loc[i]
        distances = []
        for x in range(len(df2)):
            lon2 = df2['lon'].loc[x]
            lat2 = df2['lat'].loc[x]
            dist = haversine(lon1, lat1, lon2, lat2)
            distances.append((round(dist,3),df1[name_1].loc[i],df2[name_2].loc[x]))
        min_distances.append(min(distances))
    return(min_distances)

#%% Make and test conn and cursor
from sqlalchemy import create_engine
database='DBSCAN_results'
conn = psycopg2.connect(host="localhost",database=database)
c = conn.cursor()
if c:
    print('Connection to {} is good.'.format(database))
else:
    print('Error connecting.')
c.close()

user = 'patrickmaus'
host = 'localhost'
port = '5432'
engine = create_engine('postgresql://{}@{}:{}/{}'.format(user, host, port, database))
engine

#%%
def df_to_table_with_geom(df, name, eps, min_samples):
    # add the eps and min_samples value to table name
    new_table_name = ('dbscan_results_' + name + '_' + 
                      str(eps_km).replace('.','_') + '_' + str(min_samples))
    
    # drop table if an old one exists
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS {}""".format(new_table_name))
    conn.commit()
    c.close()
    # make a new table with the df
    df.to_sql(new_table_name, engine)
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
# get data
df_full = pd.read_csv('dbscan_data.csv')
df_full.rename({'Unnamed: 0':'id'}, axis=1,inplace=True)

# sample of one ship
df_rick = df_full[df_full['mmsi']==538090091].reset_index(drop=True)


# make a df with just port activity
df_ports = df_full[df_full['port_id'] > 0]


ports_full = pd.read_csv('wpi.csv')
ports = ports_full[['index_no','port_name','latitude','longitude']]
ports = ports.rename(columns={'latitude':'lat','longitude':'lon'
                              ,'index_no':'port_id'})


#%% 


rollup_list = []
df_final_rollup = pd.DataFrame()
    

#%% set dbscan parameters
#epsilons = [.05, .1, .2, .5, .7, 1, 2, 5, 7, 10]
#samples = [10, 12, 15, 17, 20, 25, 30, 35, 40, 50, 75, 100, 150, 200, 
#           250, 300, 350, 400, 500, 1000, 1500, 2000]

epsilons = [2]
samples = [250] 

for e in epsilons:
    for s in samples:

        #this formulation will yield epsilon based on km desired
        kms_per_radian = 6371.0088
        eps_km = e
        eps = eps_km / kms_per_radian
        
        #number of min samples for core sample
        min_samples = s
        
        X = (np.radians(df_full.iloc[:100].loc[:,['lon','lat']].values))
        x_id = df_full.iloc[:100].loc[:,'id'].values
        
        tick = datetime.datetime.now()
        print("""Starting processing on {} samples with 
              eps_km={} and min_samples={} at: """.format(str(len(X)), str(eps_km), 
              str(min_samples)), tick)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', 
                        metric='haversine')
        dbscan.fit(X)
        
        print('Number of unique labels: ', len(np.unique(dbscan.labels_)))
        print('Number of  Core Samples:' , len(dbscan.core_sample_indices_))
        
        tock = datetime.datetime.now()
        lapse = tock - tick
        print('DBSCAN complete.')
        print ('Time elapsed: {}'.format(lapse))
        
        results_dict = {'id':x_id,'clust_id': dbscan.labels_,
                        'lon':np.degrees(X[:, 0]),'lat':np.degrees(X[:,1])}
        

        df_results = pd.DataFrame(results_dict)

        #determine the cluster center point, and find the distance to nearest port
        print('starting distance calculations... ')
        # group the results from the haversine by mean to get the centerpoint of the cluster
        centers = (df_results[['clust_id', 'lat','lon']]
                   .groupby('clust_id')
                   .mean()
                   .reset_index())
        
        # group the same results by count to get the total number of positions
        counts = (df_results[['clust_id', 'lat','lon']]
                  .groupby('clust_id')
                  .count())
        
        # select only one column, in this case I chose lat
        counts['counts'] = counts['lat']
        # drop the other columns so count is now just the clust_id and the summed counts
        counts.drop(['lat','lon'], axis=1, inplace=True)
        # merge counts and centers
        centers = pd.merge(centers, counts, how='left', on='clust_id')
        
        #determine distances
        dist = determine_min_distances(centers,'clust_id',ports,'port_id')
        df_dist = pd.DataFrame(dist, columns=['nearest_port_dist_from_center', 'clust_id', 
                                              'nearest_port_id'])
        
        # merge the full centers file with the results of the haversine equation
        df_summary = pd.merge(centers, df_dist, how='left', on='clust_id')
        df_summary = pd.merge(df_summary, ports[['port_name','port_id']], how='left', 
                              left_on='nearest_port_id', right_on='port_id').drop('port_id', axis=1)
        df_summary = df_summary.rename({'port_name':'nearest_port_name',
                                        'lat':'average_lat', 'lon':'average_lon'}, axis=1)
        
        #find the average distance from the centerpoint
        #distances = haversine(df_results['lon'].values, df_results['lat'].values, 33.87960, -78.49099)
        
        haver_list = []
        for i in df_summary['clust_id']:
            X = (np.radians(df_results[df_results['clust_id']==i]
                            .loc[:,['lat','lon']].values))
            Y = (np.radians(df_summary[df_summary['clust_id']==i]
                            .loc[:,['average_lat','average_lon']].values))
            haver_result = (haversine_distances(X,Y)) * kms_per_radian
            haver_dict = {'clust_id': i, 'min_dist_from_center': haver_result.min(), 
                          'average_dist_from_center':np.mean(haver_result)}
            haver_list.append(haver_dict)
            
        haver_df = pd.DataFrame(haver_list)
        
        df_summary = pd.merge(df_summary, haver_df, how='left', on='clust_id')
            
        
        #Look at how many points are designated as near ports in the database
        print('starting purity calculations...')
        df_purity = pd.merge(df_results[['clust_id','id']], 
                             df_ports[['id', 'port_name', 'port_id']], 
                             how='left', on='id')
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
        
        # roll up 
        df_rollup = pd.merge(df_purity_grouped_top, df_summary, how='left', on='clust_id')
        
        df_rollup.to_csv('./rollups/rollup_{}_{}.csv'.format(str(eps_km), str(min_samples)))
        
        tock = datetime.datetime.now()
        lapse = tock - tick
        print('All processing for this run complete.')
        print ('Time elapsed: {}'.format(lapse))
        rollup_dict = {'eps_km':eps_km, 'min_samples':min_samples, 'time':lapse, 
                        'numb_obs':len(df_results), 
                        'average_cluster_count':np.mean(df_rollup['counts'].iloc[1:]),
                        'average_ports_per_cluster':np.mean(df_rollup['counts_per_port'].iloc[1:]),
                        'average_nearest_port_from_center':np.mean(df_rollup['nearest_port_dist_from_center'].iloc[1:]),
                        'average_cluster_density':np.mean(df_rollup['average_dist_from_center'].iloc[1:]),
                        'numb_clusters':len(np.unique(dbscan.labels_)),
                        'ports_with_most_points':df_rollup['port_name_with_most_points'].iloc[1:].to_list()}
        rollup_list.append(rollup_dict)
        
        df_to_table_with_geom(df_results, 'full', eps_km, min_samples)
        
        print('Finished with round ', len(rollup_list))
        print('')
#%%
final_df = pd.DataFrame(rollup_list)
final_df.to_csv('./rollups/summary.csv')
