import numpy as np
import pandas as pd
import networkx as nx

import powerlaw

# plotting
import matplotlib.pyplot as plt
from matplotlib import colors

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

# reload modules when making edits
from importlib import reload

reload(gsta)
# %%
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.colone_cargo_params)
# %% get edgelist from database
df = gsta.get_edgelist(edge_table='cargo_edgelist_1km', engine=loc_engine, loiter_time=2)
g = nx.from_edgelist(df[['Source', 'Target']].values)
#%% get a list of the graphs to compare
kms = [1,2,3,5]
hrs = [2,4,6,8,10,12,14,16,18,20,22,24]
stats_dict = dict()
df_dict = dict()
common_nodes = set()
for km in kms:
    for hr in hrs:
        # get the dataframe
        df = gsta.get_edgelist(edge_table=f'cargo_edgelist_{km}km', engine=loc_engine, loiter_time=hr)
        # convert to graph
        g = nx.from_edgelist(df[['Source', 'Target']].values)
        # get all nodes from the graph and find the common nodes between this graph and all other graphs
        nodes = list(g.nodes())
        if common_nodes == set():
            common_nodes = set(nodes)
        else:
            common_nodes = set(common_nodes).intersection(nodes)

        # get metrics for the dataframe
        node_degree = nx.degree_centrality(g)
        degree = sum(node_degree.values()) / float(len(node_degree))
        node_betweenness = nx.betweenness_centrality(g)
        betweenness = sum(node_betweenness.values()) / float(len(node_betweenness))
        node_centrality = nx.closeness_centrality(g)
        centrality = sum(node_centrality.values()) / float(len(node_centrality))

        stats_dict[f'{km}kms_{str(hr).zfill(2)}hrs'] = {'km': km,
                                                        'hrs': hr,
                                                        'numb_nodes': len(list(g.nodes())),
                                                        'numb_edges': len(list(g.edges())),
                                                        'avg_degree': degree,
                                                        'avg_betweenness': betweenness,
                                                        'avg_centrality': centrality}

        df_dict[f'{km}kms_{str(hr).zfill(2)}hrs'] = df

df_stats = pd.DataFrame.from_dict(stats_dict, orient='index')

#%%
colors = ['k','b','y','g','r','c']
fig, ax = plt.subplots()
for km in df_stats['km'].unique():
    ax.scatter(x=df_stats[df_stats['km']==km].hrs,
               y=df_stats[df_stats['km'] == km].numb_edges,
               s=df_stats[df_stats['km'] == km].numb_nodes,
               label=km, c=colors[int(km)])
plt.xlabel('Hours')
plt.legend(bbox_to_anchor=(1, 1), ncol=1, title='Minimum km')
plt.title('Average Degree')
plt.suptitle('Relative size mapped to total number of nodes.', y=.03)
plt.show()

#%% use smallest graph as the baseline.
# get the name of the df with the smallest numb of nodes
min_size = df_stats['numb_nodes'].idxmin()
smallest_df = df_dict['1kms_02hrs']
# convert the smallest df to a graph
smallest_g = nx.from_edgelist(smallest_df[['Source', 'Target']].values)
smallest_g_common = smallest_g.subgraph(common_nodes)
# use the smallest graphs adjacency matrix as a reference point
smallest_adj = nx.to_numpy_array(smallest_g_common)

#%% compute euclid distance of each network from smallest using difference of adj tables
dist_dict = dict()
for k,v in df_dict.items():
    # convert from df to graph
    g = nx.from_edgelist(v[['Source', 'Target']].values)
    # use only the nodes common to the smallest network
    sub_g = g.subgraph(common_nodes)
    # convert to adjacency matrix
    adj = nx.to_numpy_array(sub_g)
    # compute euclid distance between two matrices
    dist = np.linalg.norm(adj - smallest_adj)
    print(dist)

    dist_dict[k] = dist

#%%
df_stats['euclid_dist'] = df_stats.index.map(dist_dict)

colors = ['k','b','y','g','r','c']
fig, ax = plt.subplots()
for km in df_stats['km'].unique():
    ax.scatter(x=df_stats[df_stats['km']==km].hrs,
               y=df_stats[df_stats['km'] == km].euclid_dist,
               s=df_stats[df_stats['km'] == km].numb_nodes,
               label=km, c=colors[int(km)])
plt.xlabel('Hours')
plt.legend(bbox_to_anchor=(1, 1), ncol=1, title='Minimum km')
plt.title('Euclid Distance of Adjacency Table Differences')
plt.suptitle('Relative size mapped to total number of nodes.', y=.03)
plt.show()

#%%
# %% get edgelist from database
df_1km_2h = gsta.get_edgelist(edge_table='cargo_edgelist_1km', engine=loc_engine, loiter_time=2)
g_1km_2h = nx.from_edgelist(df_1km_2h[['Source', 'Target']].values)

df_5km_12h = gsta.get_edgelist(edge_table='cargo_edgelist_5km', engine=loc_engine, loiter_time=12)
g_5km_12h = nx.from_edgelist(df_5km_12h[['Source', 'Target']].values)

#%%
from grakel import GraphKernel
from grakel import GraphletSampling
import grakel

gs_kernel = GraphletSampling(n_jobs=-1, normalize=True)

grakel_1km_2h = grakel.Graph(g_1km_2h.edges())
gs_kernel.fit_transform([grakel_1km_2h])
#%%
grakel_5km_12h = grakel.Graph(g_5km_12h.edges())
gs_kernel.fit_transform([grakel_5km_12h])


#%%

from LiftSRW import lift as lt
lift = lt.Lift(g_5km_12h.edges(), k=3)

