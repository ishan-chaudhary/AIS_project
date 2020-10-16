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

# Geo-Spatial Temporal Analysis package
import gsta

aws_conn = gsta.connect_psycopg2(gsta.aws_ais_cluster_params)
loc_conn = gsta.connect_psycopg2(gsta.loc_cargo_params)
aws_conn.close()    
loc_conn.close()

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
        nearest_dict ={'nearest_site_id':ports.iloc[ind[0][0]].loc['port_id'],
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
                      'max_dist_from_center': haver_result.max(),
                      'average_dist_from_center':np.mean(haver_result)}
        haver_list.append(haver_dict)
        
    haver_df = pd.DataFrame(haver_list)
    
    df_centers = pd.merge(df_centers, haver_df, how='left', on='clust_id')
    
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
    
    # create "counts at port" column through groupby
    df_purity_grouped = (df_purity
                         .groupby(['clust_id', 'port_id','port_name'])
                         .count()
                         .reset_index()
                         .rename({'id':'counts_at_port'}, axis=1))
    # create "total cluster count" column through groupby
    clust_size = (df_purity[['id','clust_id']]
                  .groupby('clust_id')
                  .count()
                  .reset_index()
                  .rename({'id':'total_clust_count'}, axis=1))
    # group the dfs back so we have all columns
    df_purity_grouped = pd.merge(df_purity_grouped, clust_size, 
                                 how='left', on='clust_id')
    # get the proportion of events near the top port
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

def composition_calc(table):
    """
    This function will get data about the mmsis that compose each cluster.
    the first quey gets the counts by mmsi per each cluster.  
    the second query will concat the strings of the mmsi name together and
    return the total number of mmsis.
    The final df will also contain the top mmsi by count and with its proportion.
    """
    get_counts_sql = """select clust_id, count(posit.mmsi) as mmsi_count
                from {} as results
                join 
                (select id, mmsi from ship_position_sample) as posit 
                on (results.id=posit.id)
                where clust_id >= 0
                group by (clust_id, posit.mmsi)
                order by clust_id, mmsi_count;""".format(table)
    count_results = pd.read_sql_query(get_counts_sql, loc_engine)
    
    get_mmsi_sql = """select clust_id, 
    string_agg(distinct(posit.mmsi), ',') as mmsis,
    count(distinct(posit.mmsi)) as mmsi_per_clust
    from {} as results
    join 
    (select id, mmsi from ship_position_sample) as posit 
    on (results.id=posit.id)
    group by (clust_id)
    order by clust_id;""".format(table)
    mmsi_results = pd.read_sql_query(get_mmsi_sql, loc_engine)
    # group by clustr id to get the total number of events within each cluster.
    grouped_comp = count_results.groupby('clust_id').sum()
    # now find the top mmsi count by dropping any duplicates but the last one.
    # since the sql quert oders by mmsi count, the last count will be the greatest.
    grouped_comp['top_mmsi_count'] = (count_results
                                      .drop_duplicates(subset='clust_id',keep='last')
                                      .reset_index()
                                      ['mmsi_count'])
    # determine the top mmsi proportiton
    grouped_comp['top_mmsi_prop'] = (grouped_comp['top_mmsi_count']/
                                     grouped_comp['mmsi_count'])
    # and finally merge with the mmsi results.
    grouped_comp = pd.merge(grouped_comp, mmsi_results, left_index=True, 
                            right_on='clust_id')
    
    return grouped_comp

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
df_all_results = pd.DataFrame()

path = '/Users/patrickmaus/Documents/projects/AIS_project/Clustering/rollups/{}/'.format(str(datetime.datetime.now().date()))
if not os.path.exists(path):
    os.makedirs(path)

epsilons = [20, 25, 30]
samples = [2500, 3000, 4000, 5000]
for e in epsilons:
    for s in samples:      
        print("""Starting analyzing clustering_analysis results with eps_km={} and min_samples={} """.format(str(e), str(s)))
        tick = datetime.datetime.now()
        
        table = 'dbscan_results_{}_{}'.format(str(e), str(s))
        df_results = pd.read_sql(table, loc_engine, columns=['id', 'lat','lon','clust_id'])
        
        #determine the cluster center point, and find the distance to nearest port
        print('Starting distance calculations... ')
        df_centers = center_calc(df_results)
        print('Finished distance calculations. ')
        
        #Look at how many points are designated as near ports in the database
        print('Starting purity calculations...')        
        df_purity = purity_calc(df_results)
        print('Finished purity calculations. ')
        
        #return details about the composition of mmsis in each cluster
        print('Starting compostion calculations...')        
        df_composition = composition_calc(table)
        print('Finished composition calculations. ')

        # roll up 
        df_rollup = pd.merge(df_purity, df_centers, how='left', on='clust_id')
        df_rollup = pd.merge(df_rollup, df_composition, how='left', on='clust_id')
        df_rollup['eps_km'] = e
        df_rollup['s'] = s
        
        df_rollup.to_csv(path+'rollup_full_{}_{}.csv'.format(str(e), str(s)))
        df_all_results = df_all_results.append(df_rollup)
        
        #timekeeping
        tock = datetime.datetime.now()
        lapse = tock - tick
        print('All processing for this run complete.')
        print ('Time elapsed: {}'.format(lapse))
        
        # the rollup dict contains multiple different metrics options.
        # non-daignostic metrics are commented out but not deleted as they 
        # may be helpful in other problems.
        rollup_dict = {'eps_km':e, 'min_samples':s, 
                       'params' : (str(e) + '_' + str(s)),
                        #built for time tracking.  not required any longer
                        #'time':lapse,
                        # number of clusters in the run
                        'numb_clusters':len(np.unique(df_rollup['clust_id'])),
                        # average positions per each cluster in each run
                        'average_cluster_count':np.mean(df_rollup['total_clust_count']),
                        
                        # distance metrics
                        # closest port.  Closer the better.
                        'average_nearest_port_from_center':np.mean(df_rollup['nearest_port_dist']),
                        # average and max distance from cluster center.
                        'average_dist_from_center':np.mean(df_rollup['average_dist_from_center']),
                        'average_max_dist_from_center':np.mean(df_rollup['max_dist_from_center']),
                        
                        # purity metrics
                        # the proporition of clusters where the most points labeled near a port
                        # a higher proportion suggests more accuracy.
                        'prop_where_most_points_labeled_as_in_ports': 1-(df_rollup['port_name_with_most_points']
                                                              .value_counts()['NONE']
                                                              /len(df_rollup)),
                        # count of clusters where most points are labeled as in port
                        'clust_numb_where_most_points_labeled_as_in_ports': (
                            df_rollup['port_name_with_most_points'].count() -
                            df_rollup['port_name_with_most_points'].value_counts()['NONE']),
                        # if this is less than one, means more than one port is near this cluster.
                       # 'average_ports_per_cluster':np.mean(df_rollup['counts_per_port']),
                        # how many positions are labeled in port.
                       # 'average_counts_per_port':np.mean(df_rollup['counts_at_port']), 
                        # proportion near top port.  Closer to 1, more homogenous.
                       # 'average_prop_per_port':np.mean(df_rollup['proportion_near_top_port']),
                        
                        # composition metrics
                        # the average proportion of the positions in a clusters made by top mmsi.
                        # higher indicate homogenity.
                      #  'average_top_mmsi_prop':np.mean(df_rollup['top_mmsi_prop']),
                        # the proportion where the top mmsi in a cluster is more than 95% of all points.  
                        # more hetrogenous clusters (less pure) could be helpful in idenitfying areas where many
                        # ships are present.
                      #  'prop_were_top_mmsi >95%': len(df_rollup[df_rollup['top_mmsi_prop']>.95])/len(df_rollup),
                        # average number of mmsis per each cluster per run
                        'average_mmsi_per_clust':(np.mean(df_rollup['mmsi_per_clust']))}
       
        rollup_list.append(rollup_dict)
        
        #df_to_table_with_geom(df_rollup, 'rollup', e, s, loc_conn)
        
        print('Finished with round ', len(rollup_list))
        print('')
#%% Make the final_df from the rollup and save to csv.
final_df = pd.DataFrame(rollup_list).round(3)
final_df['params'] = (final_df['eps_km'].astype('str') + '_' 
                      + final_df['min_samples'].astype('str'))
final_df.set_index('params', inplace=True)
final_df.to_csv(path+'summary_5k.csv')


