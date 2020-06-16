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

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)


def get_edgelist(edge_table, engine, loiter_time=2):
    # select all edges from the database and join them with the port info from wpi
    # if the node is greater than 0 (not 0 which is open ocean or null)
    # and has a time diff less than 2 hours.  That should also eliminate ports a
    # ship transits through but does not actually stop at.
    # df_stops is a table of all ports where a ship was within 5km for more than 2 hours.
    # these are the "stops" we will use to build our edgelist.
    df_stops = pd.read_sql_query(f"""select edge.node, edge.arrival_time, 
                                 edge.depart_time, edge.time_diff,
                                 edge.destination, edge.position_count, edge.mmsi, 
                                 wpi.port_name
                                 from {edge_table} as edge, wpi as wpi
                                 where edge.node=wpi.index_no and
                                 edge.node > 0 and
                                 time_diff > '{str(loiter_time)} hours';""", engine)
    df_stops.sort_values(['mmsi', 'arrival_time'], inplace=True)

    # to build the edge list, we will take the pieces from stops for the current node and the next node
    df_list = pd.concat([df_stops.node, df_stops.port_name,
                         df_stops.node.shift(-1), df_stops.port_name.shift(-1),
                         df_stops.mmsi, df_stops.mmsi.shift(-1),
                         df_stops.depart_time, df_stops.arrival_time.shift(-1)], axis=1)
    # rename the columns
    df_list.columns = ['Source_id', 'Source', 'Target_id', 'Target',
                       'mmsi', 'target_mmsi', 'source_depart', 'target_arrival']
    # drop any row where the mmsi is not the same.
    # this will leave only the rows with at least 2 nodes with valid stops, making one valid edge.
    # The resulting df is the full edge list
    df_list = (df_list[df_list['mmsi'] == df_list['target_mmsi']]
               .drop('target_mmsi', axis=1))
    # this filters ou self-loops
    df_edgelist_full = df_list[df_list['Source_id'] != df_list['Target_id']]
    return df_edgelist_full


df_edgelist = get_edgelist(edge_table='cargo_edgelist', engine=loc_engine, loiter_time=2)


# %% This produces as df that is the summarized edge list with weights
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

# %% mmsi plot

def plot_mmsi(mmsi, df_edgelist):
    mmsi_edgelist = df_edgelist[df_edgelist['mmsi'] == mmsi].reset_index(drop=True)
    mmsi_edgelist = mmsi_edgelist[['Source', 'source_depart', 'Target', 'target_arrival']]

    # build the graph
    G = nx.from_pandas_edgelist(mmsi_edgelist, source='Source', target='Target',
                                edge_attr=True, create_using=nx.MultiDiGraph)
    # get positions for all nodes
    pos = nx.spring_layout(G)

    # draw the network
    plt.figure(figsize=(6, 6))
    # draw nodes
    nx.draw_networkx_nodes(G, pos)
    # draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # add a buffer to the x margin to keep labels from being printed out of margin
    x_values, y_values = zip(*pos.values())
    x_max = max(x_values)
    x_min = min(x_values)
    x_margin = (x_max - x_min) * 0.25
    plt.xlim(x_min - x_margin, x_max + x_margin)

    # plot the title and turn off the axis
    plt.title(f'Network Plot for MMSI {str(mmsi).title()}')
    plt.axis('off')
    plt.show()

    print(mmsi_edgelist)


mmsi = '230185000'
plot_mmsi(mmsi, df_edgelist)

sample_mmsi = df_edgelist['mmsi'].sample().values[0]
print(sample_mmsi)
plot_mmsi(sample_mmsi, df_edgelist)


# %% individual source node and all related targets plot

def plot_from_source(source, df):
    # create the figure plot
    plt.figure(figsize=(8, 6))
    # get a df with just the source port as 'Source'
    df_g = df[df['Source'] == source.upper()]  # use upper to make sure fits df
    # build the network
    G = nx.from_pandas_edgelist(df_g, source='Source',
                                target='Target', edge_attr='weight',
                                create_using=nx.MultiDiGraph)
    # get positions for all nodes
    pos = nx.spring_layout(G)
    # adjust the node lable position up by .1 so self loop labels are separate
    node_label_pos = {}
    for k, v in pos.items():
        node_label_pos[k] = np.array([v[0], v[1] + .1])
    # get edge weights as dictionary
    weights = [i['weight'] for i in dict(G.edges).values()]
    #  draw nodes
    nx.draw_networkx_nodes(G, pos)
    # draw node labels
    nx.draw_networkx_labels(G, node_label_pos, font_size=10, font_family='sans-serif')
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=weights)
    # plot the title and turn off the axis
    plt.title('Weighted Network Plot for {} Port as Source'.format(source.title()),
              fontsize=16)
    plt.axis('off')
    # make a test boc for the weights.  Can be plotted on edge, but crowded
    box_str = 'Weights out from {}: \n'.format(source.title())
    for neighbor, values in G[source.upper()].items():
        box_str = (box_str + neighbor.title() + ' - ' +
                   str(values[0]['weight']) + '\n')
    plt.text(-1, -1, box_str, fontsize=12,
             verticalalignment='top', horizontalalignment='left')
    # plt.savefig("weighted_graph.png") # save as png
    plt.show()  # display

# %% plot a randomly selected source port
sample_source = df_edgelist['Source'].sample().values[0]
print(sample_source)
plot_from_source(sample_source, df_edgelist_weighted)

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

# %% Plot the whole network
plt.figure(figsize=(10, 10))
G = nx.from_pandas_edgelist(df_edgelist_weighted, source='Source',
                            target='Target', edge_attr=True,
                            create_using=nx.DiGraph)

edges = G.edges()
weights = [np.log((G[u][v]['weight'])+.1) for u,v in edges]
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

# %% Build Markov chain
markov = {}
for port in G.nodes:
    total = 0
    port_markov = {}
    for n in G[port]:
        total = total + (G[port][n]['weight'])
    for n in G[port]:
        port_markov[n] = (G[port][n]['weight'] / total)
    markov[port] = port_markov

print(markov)
df_markov = pd.DataFrame(markov)

#%%

def run_markov(start_port):
    start_port = start_port.upper()
    next_port = np.random.choice(list(markov[start_port].keys()),
                                 p=list(markov[start_port].values()))
    return next_port

markov['NEW ORLEANS']

#%%
first_port = 'NEWARK'
target_port = 'PORT EVERGLADES'
target_hop_counter = []

while len(target_hop_counter) < 10000:
    port_chain = []
    for i in range(10000):
        start_port = first_port
        try:
            next_port = np.random.choice(list(markov[start_port].keys()),
                                         p=list(markov[start_port].values()))
            port_chain.append(next_port)
            if next_port == target_port:
                target_hop_counter.append(len(port_chain))
                break
        except:
            next_port = 'chain_broken'
            port_chain.append(next_port)
            break
        start_port = next_port

        print(len(target_hop_counter))

print(target_hop_counter)

#%% plot the counts it took to get to the target port

fig, ax = plt.subplots(figsize=(8,6))
ax.hist(target_hop_counter, bins=100)
plt.title(f"Distribution of {len(target_hop_counter)} runs from {first_port.title()} to {target_port.title()}")
plt.show()


