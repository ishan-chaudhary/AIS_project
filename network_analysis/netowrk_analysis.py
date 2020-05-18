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

df_edges = pd.read_sql('cargo_edges', loc_engine)
#%%
df_edges.to_csv('cargo_ship_edgelist.csv')
#%% edge analysis
sns.distplot(df_edges['position_count'])
plt.show()
sns.distplot(df_edges['time_delta'])
plt.show()
sns.distplot(df_edges['origin'])
sns.distplot(df_edges['destination'])
plt.show()
#%%
sns.pairplot(df_edges, vars=['position_count', 'time_delta'])

#%%
G = nx.from_pandas_edgelist(df_edges, source='origin', target='destination',
                             edge_attr=True)

#%%
nx.draw(G)
#%%
nx.draw_networkx(G)

#%%

