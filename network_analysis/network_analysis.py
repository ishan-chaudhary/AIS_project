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
# %%
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.colone_cargo_params)
# %% get edgelist from database

df_edgelist = gsta.get_edgelist(edge_table='cargo_edgelist_3km', engine=loc_engine, loiter_time=8)
print(f"{len(df_edgelist)} edges and {len(df_edgelist['Source'].unique())} nodes." )
df_edgelist.to_csv('./network_analysis/edgelist.csv', index=False)
# %% This produces a df that is the summarized edge list with weights
# for the numbers of a time a ship goes from the source node to the target node.
# The code executes groupby the source/target id/name, count all the rows, drop the time fields,
# rename the remaining column from uid to weight, and reset the index
df_edgelist_weighted = gsta.get_weighted_edgelist(df_edgelist=df_edgelist)

print(f"{len(df_edgelist_weighted)} edges and {len(df_edgelist_weighted['Source'].unique())} nodes." )
#%%
sites = gsta.get_sites(loc_engine)

sites.to_csv('sites_with_regions.csv', index=False)
# %% test uid plot function
uid = '636016432'
gsta.plot_uid(uid, df_edgelist)

# %% explore random uid plots
sample_uid = df_edgelist['uid'].sample().values[0]
print(sample_uid)
gsta.plot_uid(sample_uid, df_edgelist)

# %% Plot all nodes from Boston
gsta.plot_from_source("HAMILTON", df_edgelist_weighted)

# %% plot a randomly selected source port
sample_source = df_edgelist['Source'].sample().values[0]
print(sample_source)
gsta.plot_from_source(sample_source, df_edgelist_weighted)

# %% Build report
loiter_time_hrs = 8
df_edgelist = gsta.get_edgelist(edge_table='cargo_edgelist_3km', engine=loc_engine, loiter_time=loiter_time_hrs)
df_edgelist_weighted = gsta.get_weighted_edgelist(df_edgelist=df_edgelist)
df_edgelist_weighted['Target_id'] = df_edgelist_weighted['Target_id'].astype('int')
G = nx.from_pandas_edgelist(df_edgelist_weighted, source='Source',
                            target='Target', edge_attr=True,
                            create_using=nx.DiGraph)
df_edgelist_weighted.to_csv('network_report_8_hour_loiter.csv', index=False)
#%%
G = nx.karate_club_graph()


#%%
df_report = pd.DataFrame([list(G),
                          list(G.degree(node) for node in G.nodes()),
                          list(nx.degree_centrality(G).values()),
                          list(nx.closeness_centrality(G).values()),
                          list(nx.betweenness_centrality(G).values()),
                          list(nx.clustering(G).values())]
                         ).T
df_report.columns = ['Node', 'Degree', 'Degree_Centrality',
                     'Centrality', 'Betweenness', 'Clustering_Coeff']
df_report = df_report.astype({'Degree': 'float', 'Degree_Centrality': 'float', 'Centrality': 'float',
                              'Betweenness': 'float', 'Clustering_Coeff': 'float'}).round(3)


print(df_report)
df_report.sort_values('Degree',inplace=True, ascending=False)
df_report.to_csv('report_sorted_degree.csv', index=False)

print(f"""
The networks has {len(G.nodes)} nodes and {len(G.edges)} edges.
Highest degree node is {df_report['Node'].loc[df_report['Degree'].idxmax()]}.
Highest centrality node is {df_report['Node'].loc[df_report['Centrality'].idxmax()]}
Highest betweenness node is {df_report['Node'].loc[df_report['Betweenness'].idxmax()]}.
Highest Clustering Coeff node is {df_report['Node'].loc[df_report['Clustering_Coeff'].idxmax()]}.
""")

#%% build stats df for a given G
km = 3
hr = 8
stats_dict = dict()

# get metrics for the dataframe
node_degree = nx.degree_centrality(G)
degree = sum(node_degree.values()) / float(len(node_degree))
node_betweenness = nx.betweenness_centrality(G)
betweenness = sum(node_betweenness.values()) / float(len(node_betweenness))
node_centrality = nx.closeness_centrality(G)
centrality = sum(node_centrality.values()) / float(len(node_centrality))
node_clustering_coeff = nx.clustering(G)
clustering_coeff = sum(node_clustering_coeff.values()) / float(len(node_clustering_coeff))
transitivity = nx.transitivity(G)

stats_dict[f'summary'] = {'km': km,
                        'hrs': hr,
                        'numb_nodes': len(list(G.nodes())),
                        'numb_edges': len(list(G.edges())),
                        'avg_degree': degree,
                        'avg_betweenness': betweenness,
                        'avg_centrality': centrality,
                        'avg_clust_coeff': clustering_coeff,
                        'transitivity': transitivity}

df_stats = pd.DataFrame.from_dict(stats_dict, orient='index')
df_stats.round(3).to_csv('summary_stats.csv', index=False)

# %%
df_report[['Degree', 'Centrality', 'Betweenness', 'Clustering_Coeff']].hist()
plt.show()

df_report.boxplot(['Degree_Centrality', 'Centrality',
                   'Betweenness', 'Clustering_Coeff'])
plt.title('AIS Network with 3km and 8 hours as Hyperparameters')
plt.show()

#%%
df_report.boxplot(['Degree'])
plt.show()



#%%
import collections
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram for AIS Network with 3km and 8hr Loiter")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.show()

#%%
import powerlaw

data = [x[1] for x in list(nx.degree(G))]
fit = powerlaw.Fit(data)
print(f"Powerlaw coeffecient from fit: {fit.power_law.alpha}")
# R is the Loglikelihood ratio of the two distributions' fit to the data. If greater than 0,
# the first distribution is preferred. If less than 0, the second distribution is preferred.
# P is the significance of R.
R, p = fit.distribution_compare('power_law', 'lognormal')
print(f"Logliklihood ratio: {R} with a p-value of {p}.")
powerlaw.plot_pdf(data, linear_bins=False)
plt.title(f"AIS Data Power Law Plot with Coefficent of {round(results.power_law.alpha,3)}")
plt.show()


fig = fit.plot_ccdf(label='Emprical Data')
fit.power_law.plot_ccdf(ax=fig, color='r', linestyle='--', label='Power law fit')
fit.lognormal.plot_ccdf(ax=fig, color='g', linestyle='--', label='Lognormal fit')
handles, labels = fig.get_legend_handles_labels()
fig.legend(handles, labels, loc=3)
plt.title(f"AIS Data CCDF Plot")

plt.show()




# %% Plot the whole network.  Doesnt work well in networkx, recommend using gephi
plt.figure(figsize=(10, 10))


edges = G.edges()
#weights = [np.log((G[u][v]['weight']) + .1) for u, v in edges]
pos = nx.spring_layout(G)  # positions for all nodes
# nodes
nx.draw_networkx_nodes(G, pos)
# edges
nx.draw_networkx_edges(G, pos)
# labels
nx.draw_networkx_labels(G, pos)
plt.axis('off')
plt.title('Full Network Plot')
plt.show()

