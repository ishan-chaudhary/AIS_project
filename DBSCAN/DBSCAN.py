#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#time tracking
import datetime

# db admin
import psycopg2
from sqlalchemy import create_engine


#%% PostGres Connections

database='ais_test'
engine = create_engine('postgresql://patrickmaus@localhost:5432/{}'.format(database))

engine

#%%
# get data
df_full = pd.read_csv('dbscan_data.csv')

# sample of one ship
df_rick = df_full[df_full['mmsi']==538090091].reset_index(drop=True)
df_rick.info()

# make a df with just port activity
df_ports = df_full[df_full['port_id'] > 0]
df_ports.info()

ports_full = pd.read_csv('wpi.csv')
ports = ports_full[['index_no','port_name','latitude','longitude']]
ports = ports.rename(columns={'latitude':'lat','longitude':'lon'})
ports.head()

print(len((df_ports['port_id'].unique())))
print(len(df_ports))

#%% set dbscan parameters

from sklearn.cluster import DBSCAN

eps = .25
min_samples = 50

X = df_full.iloc[:100000].loc[:,['lon','lat']].values

#%%
summary_list = []
#%% run dbscan
tick = datetime.datetime.now()
print("""Starting processing on {} samples with 
      eps={} and min_samples={} at: """.format(str(len(X)), str(eps), 
      str(min_samples)), tick)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X)

print('Number of unique labels: ', len(np.unique(dbscan.labels_)))
print('Number of  Core Samples:' , len(dbscan.core_sample_indices_))

tock = datetime.datetime.now()
lapse = tock - tick
print ('Time elapsed: {} \n'.format(lapse))

results_dict = {'clust_id': dbscan.labels_,'lon':X[:, 0],'lat':X[:,1]}
summary_dict = {'eps':eps, 'min_samples':min_samples, 'time':lapse, 
                'numb_obs':len(X), 'numb_clusters':len(np.unique(dbscan.labels_))}


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
    r = 3956  # 6371 Radius of earth in kilometers. Use 3956 for miles
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

#%%

# group the results from the haversine by mean to get the centerpoint of the cluster
centers = df_results[['clust_id', 'lat','lon']].groupby('clust_id').mean().reset_index()
# group the same results by count to get the total number of positions
counts = df_results[['clust_id', 'lat','lon']].groupby('clust_id').count()
# select only one column, in this case I chose lat
counts['counts'] = counts['lat']
# drop the other columns so count is now just the clust_id and the summed counts
counts.drop(['lat','lon'], axis=1, inplace=True)
# merge counts and centers
centers = pd.merge(centers, counts, how='left', on='clust_id')

dist = determine_min_distances(centers,'clust_id',ports,'port_name')
df_dist = pd.DataFrame(dist, columns=['distance from center', 'clust_id', 'nearest_port'])

# merge the full centers file with the results of the haversine equation
df_summary = pd.merge(centers, df_dist, how='left', on='clust_id')
df_summary.head()

#%%
 ## Adding to Database for QGIS visualization


#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database=database)
c = conn.cursor()
if c:
    print('Connection to {} is good.'.format(database))
else:
    print('Error connecting.')
c.close()

#%%
def df_to_table_with_geom(df, name, eps, min_samples):
    # add the eps and min_samples value to table name
    new_table_name = 'dbscan_results_' + name + '_' + '_' + str(min_samples)
    
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
df_to_table_with_geom(df_results, 'df_rick', eps, min_samples)

