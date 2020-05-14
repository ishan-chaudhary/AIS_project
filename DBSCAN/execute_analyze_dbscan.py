#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:11:39 2020

@author: patrickmaus
"""

#time tracking
import datetime

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config


conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)

#%%

epsilons_km = [.25, .5, 1, 2, 3, 5, 7]
samples = [2, 5, 10, 25, 50, 100, 250, 500]

eps_samples_params = []
for eps_km in epsilons_km:
    for min_samples in samples: 
        eps_samples_params.append([eps_km, min_samples])
#%% assign hyperparameters to investigate
eps_samples_params = [[.5, 50],
                      [.5, 100]]        

#%%    
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
gsta.execute_dbscan('ship_position_sample', eps_samples_params, conn, loc_engine, method='sklearn_mmsi', 
                   drop_schema=True)
conn.close()

