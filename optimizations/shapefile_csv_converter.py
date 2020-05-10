#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:13:29 2020

@author: patrickmaus
"""

import os
import geopandas as gpd
#%%
## Identifying files
data_dir = 'path/to/data'
files = os.listdir(data_dir)
shp_files = [file for file in files if '.shp' in file]

## Reading in shps and saving as CSVs
new_data_dir = 'path/to/new/data'
for shp_file in shp_files:
    gdf = gpd.read_file(f'{data_dir}/{shp_file}'
    gdf.to_csv(f'{new_data_dir}/{shp_file[:-4]}.csv')
    
#%%

gdf = gpd.read_file('us_medium_shoreline/us_medium_shoreline.shp')
gdf.to_csv('us_medium_shoreline.csv')