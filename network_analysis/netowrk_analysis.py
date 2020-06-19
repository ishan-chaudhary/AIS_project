#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:15:33 2020

@author: patrickmaus
"""

import numpy as np
import pandas as pd
import networkx as nx

# plotting
import matplotlib.pyplot as plt

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config
#%%
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)
#%%
df_edgelist = gsta.get_edgelist(edge_table='cargo_edgelist', engine=loc_engine, loiter_time=2)

# %% This produces a df that is the summarized edge list with weights
# for the numbers of a time a ship goes from the source node to the target node.
# groupby the source/target id/name, count all the rows, drop the time fields,
# rename the remaining column from mmsi to weight, and reset the index
df_edgelist_weighted = (df_edgelist.groupby(['Source_id', 'Source',
                                             'Target_id', 'Target'])
                        .count()
                        .drop(['source_depart', 'target_arrival'], axis=1)
                        .rename(columns={'mmsi': 'weight'})
                        .reset_index())

print(len(df_edgelist_weighted[df_edgelist_weighted['weight'] < 2]))

df_edgelist_weighted.to_csv('edgelist_weighted.csv')

# %% mmsi plot

mmsi = '230185000'
gsta.plot_mmsi(mmsi, df_edgelist)

sample_mmsi = df_edgelist['mmsi'].sample().values[0]
print(sample_mmsi)
gsta.plot_mmsi(sample_mmsi, df_edgelist)


# %% individual source node and all related targets plot



# %% plot a randomly selected source port
sample_source = df_edgelist['Source'].sample().values[0]
print(sample_source)
gsta.plot_from_source(sample_source, df_edgelist_weighted)

#%%
gsta.plot_from_source("BOSTON", df_edgelist_weighted)
# %% Build report
G = nx.from_pandas_edgelist(df_edgelist_weighted, source='Source',
                            target='Target', edge_attr=True,
                            create_using=nx.DiGraph)

df_report = pd.DataFrame([nx.degree_centrality(G),
                          nx.in_degree_centrality(G),
                          nx.out_degree_centrality(G),
                          nx.closeness_centrality(G),
                          nx.betweenness_centrality(G),
                          nx.katz_centrality(G)]
                         ).T
df_report.columns = ['Degree', 'In-Degree', 'Out-Degree', 'Closeness Centrality', 'Betweenness',
                     'Katz Centrality']
print(df_report)
# %% Plot the whole network
plt.figure(figsize=(10, 10))
G = nx.from_pandas_edgelist(df_edgelist_weighted, source='Source',
                            target='Target', edge_attr=True,
                            create_using=nx.DiGraph)

edges = G.edges()
weights = [np.log((G[u][v]['weight'])+.1) for u, v in edges]
pos = nx.spring_layout(G)  # positions for all nodes
# nodes
nx.draw_networkx_nodes(G, pos)
# edges
nx.draw_networkx_edges(G, pos, width=weights)
# labels
nx.draw_networkx_labels(G, pos)
plt.axis('off')
plt.title('Full Network Plot')
plt.show()
