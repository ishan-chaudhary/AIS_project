#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:15:33 2020

@author: patrickmaus
"""

import numpy as np
import pandas as pd
import datetime
import networkx as nx

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config
#%%
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)

df_edges = pd.read_sql('cargo_edgelist', loc_engine)
df_edges.sort_values(['mmsi','arrival_time'], inplace=True)
#%%
#df_edges.to_csv('cargo_ship_edgelist.csv')
#%% edge analysis
sns.distplot(df_edges['position_count'])
plt.show()
sns.distplot(df_edges['time_diff'])
plt.show()
sns.distplot(df_edges['node'])
sns.distplot(df_edges['destination'])
plt.show()

#%%
df_edges_no_0 = df_edges[(df_edges['node'] != 0)]
df_edges_no_0 = df_edges_no_0[(df_edges_no_0['destination'] > 0)]

df_filtered = df_edges_no_0[(df_edges_no_0['time_diff'] > pd.Timedelta('2 hours'))] 
#df_filtered = df_filtered



#%%
G = nx.from_pandas_edgelist(df_filtered, source='node', target='destination',
                             edge_attr=True)

#%%
nx.draw(G)
#%%
nx.draw_networkx(G)

#%%

