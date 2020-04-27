#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:51:09 2020

@author: patrickmaus
"""

import pandas as pd
import numpy as np
import os

#time tracking
import datetime

#%% use this to reload data if needed
#path = '/Users/patrickmaus/Documents/projects/AIS_project/DBSCAN/rollups/2020-04-24/'
#final_df = pd.read_csv(path+'summary_5k.csv')
#%% build some scatterplots
import matplotlib.pyplot as plt
#although not used, Axes3D is required
from mpl_toolkits.mplot3d import Axes3D
def scatter_3d(value, df, highlight=None):
    final_df = df
    X = final_df['eps_km']
    Y = final_df['min_samples']
    Z = final_df[value]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # if you want to plot a particular set of hyperparameters, input its
    # index value and that parameter will be highlighted a different color.
    if highlight:
        ax.scatter(X[np.arange(len(X))!=highlight], Y[np.arange(len(Y))!=highlight], 
                    Z[np.arange(len(Z))!=highlight], c='r', marker='o')
        ax.scatter(X[highlight], Y[highlight], Z[highlight], c='black', marker='D')
    else:
        ax.scatter(X, Y, Z, c='r', marker='o')
        
    ax.set_xlabel('eps_km')
    ax.set_ylabel('min_samples')
    ax.set_zlabel(value)
    plt.title('DBSCAN Metrics Evaluation for {}'.format(value))
    ax.view_init(30, 160)
    

#%% Plot the charts
df_cols = final_df.columns
features_for_plot = np.setdiff1d(df_cols, ['eps_km','min_sample','time' ])

for f in features_for_plot:
    scatter_3d(f, final_df)


#%% Scaling
key_features = ['average_max_dist_from_center', #minimize
                'average_nearest_port_from_center', #minimize, high rank
                'average_mmsi_per_clust', #maximise
                'prop_where_most_points_labeled_as_in_ports'] #maximize, high rank

df = final_df[key_features]
df['clust_numb_where_most_points_labeled_as_in_ports'] = (final_df['numb_clusters'] *
                                                          final_df['prop_where_most_points_labeled_as_in_ports'])

from sklearn import preprocessing
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_scaled = pd.DataFrame(x_scaled, columns=df.columns, index=df.index).round(5)

# three features are minimized, 2 need to be maximize.
# 1 - minimized feature will get us the maximized feature
minimized_metrics = ['average_max_dist_from_center',
                     'average_nearest_port_from_center']
for m in minimized_metrics:
    df_scaled[m] = (1 - df_scaled[m])
    

df_scaled['average'] = df_scaled.mean(axis=1)
df_scaled['weighted'] = ((df_scaled['average_max_dist_from_center'] * .1) +
                        (df_scaled['average_nearest_port_from_center'] * .2) +
                        (df_scaled['average_mmsi_per_clust'] * .1) +
                        (df_scaled['prop_where_most_points_labeled_as_in_ports'] * .3) +
                        (df_scaled['clust_numb_where_most_points_labeled_as_in_ports'] * .3))

df_scaled['inport_combo'] = (df_scaled['prop_where_most_points_labeled_as_in_ports'] +
                             df_scaled['clust_numb_where_most_points_labeled_as_in_ports'])

print(df_scaled)
print(df)

df_scaled.round(3).to_csv('scaled_results.csv')