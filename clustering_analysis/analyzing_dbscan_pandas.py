# get positions
# get sites
# get nearest site for each position
#%%
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances


import folium
from gnact import utils, clust, network
import db_config
import warnings
warnings.filterwarnings('ignore')

# create the engine to the database
engine = utils.connect_engine(db_config.colone_cargo_params, print_verbose=True)
#%%


#%%
df_sites = clust.get_sites_wpi((engine))
df_posits = clust.get_uid_posits(('636016432',), engine, end_time='2018-01-01')
df_clusts = clust.calc_clusts(df_posits, eps_km=3, min_samp=250, method='dbscan')
df_centers = clust.calc_centers(df_clusts)
df_nn = clust.calc_nn(df_posits, df_sites)
df_nearest_sites = clust.calc_nn(df_centers, df_sites, lat='average_lat', lon='average_lon', id='clust_id')
#%%

#%%
df_edgelist = network.calc_edgelist(df_posits, df_nn, dist_km=3, loiter_time_mins=360)
df_nearby_activity = network.calc_nearby_activity(df_edgelist, df_sites)

df_stats = clust.calc_stats(df_clusts, df_sites, df_nearby_activity, dist_threshold=3)


#%%
def calc_stats(df_clusts, df_nn, df_sites, noise_filter=10, dist_threshold=3):
    # this function needs data from both the df_centers and df_nearest_sites to produce a comprehensive cluster rollup
    df_centers = calc_centers(df_clusts)
    df_nearest_sites = calc_nn(df_centers, df_sites, lat='average_lat', lon='average_lon', id='clust_id')
    df_clust_rollup = pd.merge(df_centers, df_nearest_sites, how='inner', left_on='clust_id', right_on='id')
    df_clust_rollup = df_clust_rollup[['clust_id', 'nearest_site_id', 'dist_km', 'total_clust_count']]

    # this function also needs a summary of all the sites that the raw data occurs near, grouped and filtered.
    # drop any points that are farther away than the set distance threshold, drop the id column, groupby the site_id,
    # get the agg count, and reset index.
    df_nn_grouped = (df_nn[df_nn['dist_km'] < dist_threshold]
                     .drop('id', axis=1)
                     .groupby(['nearest_site_id'])
                     .agg('count')
                     .reset_index(drop=False))
    # rename columns
    df_nn_grouped.columns = ['nearest_site_id', 'total_raw_count']
    # filter out any sites that have fewer points than defined in the noise filter
    df_nn_filtered = df_nn_grouped[df_nn_grouped['total_raw_count'] > noise_filter]

    df_stats = pd.merge(df_clust_rollup[df_clust_rollup['dist_km'] < dist_threshold], df_nn_filtered,
                        how='outer', on='nearest_site_id', indicator=True)

    # this df lists where the counts in the merge.
    # left_only are sites only in the clustering results.  (false positives for clustering results)
    # right_only are sites where the raw data was near but no cluster found within dist_treshold.  (false negatives for clustering results)
    # both are ports in both datasets.  (true positives for clustering results)
    values = (df_stats['_merge'].value_counts())
    # recall is the proporation of relevant items selected
    # it is the number of true positives divided by TP + FN
    stats_recall = values['both'] / (values['both'] + values['right_only'])
    # precision is the proportion of selected items that are relevant.
    # it is the number of true positives our of all items selected by dbscan.
    stats_precision = values['both'] / len(df_stats)
    # now find the f_measure, which is the harmonic mean of precision and recall
    stats_f_measure = 2 * ((stats_precision * stats_recall) / (stats_precision + stats_recall))
    print(f'Precision: {round(stats_precision,4)}, Recall: {round(stats_recall,4)}, '
          f'F1: {round(stats_f_measure,4)}')

    return df_stats