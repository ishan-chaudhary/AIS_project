
import numpy as np
import pandas as pd
import networkx as nx

# plotting
import matplotlib.pyplot as plt

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

# reload modules when making edits
from importlib import reload
reload(gsta)
#%%
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_full_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_full_params)
#%% get edgelist from database
df_edgelist = gsta.get_edgelist(edge_table='cargo_edgelist', engine=loc_engine, loiter_time=2)
# df_edgelist.to_csv('edgelist.csv')

# %% This produces a df that is the summarized edge list with weights
# for the numbers of a time a ship goes from the source node to the target node.
# The code executes groupby the source/target id/name, count all the rows, drop the time fields,
# rename the remaining column from uid to weight, and reset the index
df_edgelist_weighted = gsta.get_weighted_edgelist(df_edgelist=df_edgelist)

print(len(df_edgelist_weighted[df_edgelist_weighted['weight'] < 2]))
df_edgelist_weighted[df_edgelist_weighted['weight'] > 2].to_csv('edgelist_weighted.csv', index=False)

#%% test uid plot function
uid = '636016432'
gsta.plot_uid(uid, df_edgelist)

#%% explore random uid plots
sample_uid = df_edgelist['uid'].sample().values[0]
print(sample_uid)
gsta.plot_uid(sample_uid, df_edgelist)

#%% Plot all nodes from Boston
gsta.plot_from_source("Boston", df_edgelist_weighted)

# %% plot a randomly selected source port
sample_source = df_edgelist['Source'].sample().values[0]
print(sample_source)
gsta.plot_from_source(sample_source, df_edgelist_weighted)

# %% Build report
loiter_time_hrs = 2
df_edgelist = gsta.get_edgelist(edge_table='cargo_edgelist', engine=loc_engine, loiter_time=loiter_time_hrs)
df_edgelist_weighted = gsta.get_weighted_edgelist(df_edgelist=df_edgelist)
G = nx.from_pandas_edgelist(df_edgelist_weighted, source='Source',
                            target='Target', edge_attr=True,
                            create_using=nx.DiGraph)

df_report = pd.DataFrame([list(G),
                          [len(G[node]) for node in G.nodes()],
                          list(nx.degree_centrality(G).values()),
                          nx.in_degree_centrality(G).values(),
                          nx.out_degree_centrality(G).values(),
                          nx.closeness_centrality(G).values(),
                          nx.betweenness_centrality(G).values()]
                         ).T
df_report.columns = ['Node', 'Targets', 'Degree', 'In-Degree', 'Out-Degree',
                     'Centrality', 'Betweenness']
df_report = df_report.astype({'Degree':'float', 'In-Degree':'float', 'Out-Degree':'float',
                     'Centrality':'float', 'Betweenness':'float'}).round(3)

print(df_report)
df_report.round(3).to_csv(f'network_report_{loiter_time_hrs}_hour_loiter.csv', index=False)

#%%
df_report_sorted = df_report.sort_values('Closeness Centrality', ascending=False)

# %% Plot the whole network.  Doesnt work well in networkx, recommend usign gephi
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