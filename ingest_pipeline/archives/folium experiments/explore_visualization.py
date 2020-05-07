#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 00:34:48 2019

@author: patrickmaus
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
import psycopg2
from sqlalchemy import create_engine

import folium

#%% Establish connection and test
conn = psycopg2.connect(host="localhost",database="ais_data")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()
#%% Establish connection and slect a sample from the database
c = conn.cursor()
df_sample = pd.read_sql('select * from ship_position limit 1000', conn)
c.close()
#%% Plot World Map

world_gdf = gpd.read_file('../AIS_data/shapefiles/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
world_gdf = world_gdf[world_gdf['NAME'] != 'Antarctica']
ax = world_gdf.plot(color='lightgrey', edgecolor='silver', figsize=(18,10))
plt.show()
#%% Test plotting all the positions using Geopandas
positions = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in 
                                       zip(df_sample['lon'],df_sample['lat'])])
print('positions created')
g_df = pd.merge(positions, df_sample, left_index=True, right_index=True)
ax = world_gdf.plot(color='lightgrey', edgecolor='silver', figsize=(18,10))
g_df.plot(ax=ax)
print('positions plotted')
plt.show()

#%% Group all positions in the df by ship MMSI and plot using folium
df = df_sample

m = folium.Map(
    location=[0,0],
    zoom_start=2,
    tiles='Stamen Terrain'
)

df['points'] = list(zip(df.lat, df.lon))

for mmsi in (df.mmsi.unique()):
    df_m = df[df['mmsi'] == mmsi]
    points = df_m['points'].to_list()
    folium.PolyLine(points).add_to(m)
    for each in points:  
        folium.Marker(each, popup='<i>{}</i>'.format(mmsi)).add_to(m)
    print(mmsi, ' done processing.')
        
    
m.save("mymap_tracks.html")   

#%%

m = folium.Map(
    location=[0,0],
    zoom_start=2,
    tiles='Stamen Terrain'
)

for name, row in df_sample.iterrows():
    folium.Marker([row[2],row[3]], popup='<i>{}\n{}</i>'.format(row[0],row[1])).add_to(m)   

m.save("mymap.html")
