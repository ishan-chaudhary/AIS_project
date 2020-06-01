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

df_data = pd.read_sql_query("""select edge.node, edge.arrival_time, 
                             edge.depart_time, edge.time_diff,
                             edge.destination, edge.position_count, edge.mmsi, 
                             wpi.port_name
                             from cargo_edgelist as edge, wpi as wpi
                             where edge.node=wpi.index_no;""", loc_engine)
df_data.sort_values(['mmsi','arrival_time'], inplace=True)

#%%
# removed any stops that have a node of 0, which is open ocean.
# we arent going to treat the ocean as a node, but had to capture
# activity there in the earlier steps.
df_data_no_0 = df_data[(df_data['node'] != 0)]

# df_stops is a table of all ports where a ship was within 5km for more than 2 hours.
# these are the "stops" we will use to build our edgelist.
df_stops = (df_data_no_0[(df_data_no_0['time_diff'] > pd.Timedelta('2 hours'))]
            .sort_values(['mmsi','arrival_time']))
#%%
# take the pieces from stops for the current node and the next node
df_list = pd.concat([df_stops.node, df_stops.port_name, 
                     df_stops.node.shift(-1), df_stops.port_name.shift(-1), 
                     df_stops.mmsi, df_stops.mmsi.shift(-1),
                     df_stops.depart_time, df_stops.arrival_time.shift(-1)], axis=1)
# renmae the columns
df_list.columns = ['Source_id', 'Source', 'Target_id', 'Target',
                   'mmsi', 'target_mmsi', 'source_depart', 'target_arrival',]
# drop any row where the vessl id is not the same.
# this will leave only the rows with at least 2 nodes with valid stops,
# making one valid edge.  
# The resulting df is the full edge list
df_list = (df_list[df_list['mmsi']==df_list['target_mmsi']]
           .drop('target_mmsi', axis=1))
# this filters ou self-loops
df_edgelist_full = df_list[df_list['Source_id']!=df_list['Target_id']]
df_edgelist_full.to_csv('edgelist_full.csv', index=False)

#%% make a summary of ports visited for each vessel
df_trips = (df_list.reset_index(drop=True))
df_trips['Target_id'] = df_trips['Target_id'].astype('int')

df_trips['trips'] = (df_trips['Source_id'].astype('str') + ':' + 
                     df_trips['Target_id'].astype('str'))
df_trips = df_trips.drop(['source_depart', 'target_arrival', 
                          'Source', 'Target',
                          'Source_id', 'Target_id'], axis=1)
df_grouped_trips = df_trips.groupby('mmsi')['trips'].apply(list).to_frame()
df_grouped_trips['trip_lengh'] = df_grouped_trips['trips'].apply(len)


#%% This produces as df that is the summarized edge list with wieghts
# for the numbers of a time a ship goes from the source node to the target node.

# groupby the source/target id/name, count all the rows, drop the time fields,
# rename the remaining column from mmsi to weight, and rest the index
df_edgelist_weighted = (df_list.groupby(['Source_id', 'Source', 
                                'Target_id', 'Target'])
              .count()
              .drop(['source_depart', 'target_arrival'], axis=1)
              .rename(columns={'mmsi':'weight'})
              .reset_index())
#df_edgelist.to_csv('edgelistV2.csv', index=False)
#%%
mmsi = '230185000'
mmsi_edgelist = df_edgelist_full[df_edgelist_full['mmsi'] == mmsi]
G = nx.from_pandas_edgelist(mmsi_edgelist, source='Source', 
                            target='Target', edge_attr=True)

#%%
#print(len(df_edgelist_weighted[df_edgelist_weighted['weight']>1]))
plt.figure(figsize=(10,10)) 
G=nx.MultiDiGraph()
for row in (df_edgelist_weighted
            .sort_values(by='weight', ascending=False)
            .iloc[:500]
            .iterrows()):
    G.add_edge(row[1]['Source'], row[1]['Target'], weight=row[1]['weight'])
    
pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,node_size=70)

# edges
nx.draw_networkx_edges(G,pos,alpha=0.5,edge_color='b')

# labels
nx.draw_networkx_labels(G,pos,font_size=10,font_family='sans-serif')

plt.axis('off')

plt.savefig("weighted_graph.png") # save as png
plt.show() # display





#%%
plt.figure(figsize=(10,10)) 
G = nx.from_pandas_edgelist(df_edgelist_weighted, source='Source', 
                            target='Target', edge_attr=True, 
                            create_using=nx.MultiDiGraph)
print(G.number_of_edges())
print(type(G))

pos=nx.spring_layout(G) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,node_size=70)

# edges
nx.draw_networkx_edges(G,pos,alpha=0.5,edge_color='b')

# labels
nx.draw_networkx_labels(G,pos,font_size=10,font_family='sans-serif')
#%%
#print(nx.degree_centrality(G))
print(list(G.neighbors('LONG BEACH')))
#adj = nx.adjacency_matrix(G)
total = 0
markov = {}
for n in G['LONG BEACH']:
    total = total + (G['LONG BEACH'][n][0]['weight'])
for n in G['LONG BEACH']:
     markov[n] = round((G['LONG BEACH'][n][0]['weight']/total),3)
#%%
nx.draw_circular(G)
#%%
nx.draw_kamada_kawai(G)
#%%

pos = nx.circular_layout(G)
plt.show()
