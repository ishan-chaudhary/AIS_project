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

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config
#%%
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)

edge_df = pd.read_sql('cargo_edges', loc_engine)
#%%
G = nx.from_pandas_edgelist(edge_df, source='origin', target='destination',
                             edge_attr=True)

#%%
nx.draw(G)
#%%
nx.draw_networkx(G)

#%%

