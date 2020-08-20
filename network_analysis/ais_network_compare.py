import pandas as pd
import random

# network tools
import networkx as nx
import netrd
import grakel

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


# %% get a list of the graphs to compare
def get_edgelists_dict(kms, hrs, dates, engine):
    """
    Pull edgelists of the defined km, hours, and dates from the designated database.
    :param kms:
    :param hrs:
    :param dates:
    :param engine:
    :return:
    """
    dict_edgelists = dict()
    for km in kms:
        for hr in hrs:
            for start_date, end_date in dates:
                # get the dataframe
                df = gsta.get_edgelist(edge_table=f'cargo_edgelist_{km}km', engine=engine, loiter_time=hr,
                                       start_date=start_date, end_date=end_date)
                dict_edgelists[f'{km}kms_{str(hr).zfill(2)}hrs_{start_date}'] = df
    return dict_edgelists


def calc_network_stats(dict_edgelists):
    """
    Returns a df_stat with relevant stats comparing networks as well as a list of all the common nodes.
    :param df_dict:
    :return:
    """
    common_nodes = set()
    stats_dict = dict()
    for k, v in dict_edgelists.items():
        # convert to graph
        g = nx.from_edgelist(v[['Source', 'Target']].values)
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
        node_clustering_coeff = nx.clustering(g)
        clustering_coeff = sum(node_clustering_coeff.values()) / float(len(node_clustering_coeff))
        transitivity = nx.transitivity(g)

        stats_dict[k] = {'numb_nodes': len(list(g.nodes())),
                         'numb_edges': len(list(g.edges())),
                         'avg_degree': degree,
                         'avg_betweenness': betweenness,
                         'avg_centrality': centrality,
                         'avg_clust_coeff': clustering_coeff,
                         'transitivity': transitivity}

    df_stats = pd.DataFrame.from_dict(stats_dict, orient='index')
    return df_stats, common_nodes


def plot_network_params_comparisons(df, y):
    """
    For only the df_stats with km and hrs, plot a scatterplot of the y variable
    :param df:
    :param y:
    :return:
    """
    color_list = ['k', 'b', 'y', 'g', 'r', 'c']
    fig, ax = plt.subplots()
    for km in df['km'].unique():
        ax.scatter(x=df[df['km'] == km].hrs,
                   y=df[df['km'] == km][y],
                   s=df[df['km'] == km].numb_nodes,
                   label=km, c=color_list[int(km)])
    plt.xlabel('Hours')
    plt.legend(bbox_to_anchor=(1, 1), ncol=1, title='Minimum km')
    plt.title(y)
    plt.suptitle('Relative size mapped to total number of nodes.', y=.03)
    plt.show()


def get_smallest_graph(dict_edgelists, df_stats, common_nodes):
    """
    Give a dict of edgelists, the df_stats, and the list of common nodes,
    return the smallest graph with only common nodes
    :param dict_edgelists:
    :param df_stats:
    :param common_nodes:
    :return:
    """
    min_size = df_stats['numb_nodes'].idxmin()
    smallest_df = dict_edgelists[min_size]
    # convert the smallest df to a graph
    smallest_g = nx.from_edgelist(smallest_df[['Source', 'Target']].values)
    smallest_g_common = smallest_g.subgraph(common_nodes)
    return smallest_g_common


def compute_network_distances(dict_edgelists, df_stats, common_nodes, dist_metric, dist_metric_name,
                              ref_type, package):
    """
    For a given dict of edgelists, df_stats, and common nodes, use the given dist metric and name to
    compute distance between reference graph and each graph in the dict.  Update the stats df with distance.
    :param dict_edgelists:
    :param df_stats:
    :param common_nodes:
    :param dist_metric:
    :param dist_metric_name:
    :param ref_type:
    :param package:
    :return:
    """
    if ref_type == 'smallest':
        smallest_g = get_smallest_graph(dict_edgelists, df_stats, common_nodes)
        if package == 'grakel':
            converted_g = nx.convert_node_labels_to_integers(smallest_g, label_attribute='name')
            node_lables = nx.get_node_attributes(converted_g, 'name')
            smallest_grakel = grakel.Graph(converted_g.edges(), node_labels=node_lables)
            # fit and transform the model
            gk = dist_metric(normalize=True)
            gk.fit_transform([smallest_grakel])
    elif ref_type == 'previous':
        previous = None
    else:
        raise ValueError("'ref_graph' must be 'previous' or 'smallest'.")
    # check garkel or netrd is selected
    if package not in ['grakel', 'netrd']:
        raise ValueError("'package' must be 'grakel' or 'netrd'.")

    dist_dict = dict()
    for k, v in dict_edgelists.items():
        # convert from df to graph
        g = nx.from_edgelist(v[['Source', 'Target']].values)
        # use only the nodes common to the smallest network
        sub_g = g.subgraph(common_nodes)
        # compute distance based on ref type and package name.  first 'smallest' comparsion
        if ref_type == 'smallest':
            if package == 'netrd':
                dist = dist_metric.dist(G1=smallest_g, G2=sub_g)
                dist_dict[k] = dist
            if package == 'grakel':
                converted_g = nx.convert_node_labels_to_integers(g, label_attribute='name')
                node_lables = nx.get_node_attributes(converted_g, 'name')
                grakel_g = grakel.Graph(converted_g.edges(), node_labels=node_lables)
                # compute distance between two graphs using the grakel package
                result = gk.transform([grakel_g])
                dist_dict[k] = result[0][0]
        # next compute if ref type is previous, ie sequetial data
        elif ref_type == 'previous':
            if previous is None:
                if package == 'netrd':
                    previous = sub_g
                if package == 'grakel':
                    # convert the g we are iterating over to continuous
                    converted_g = nx.convert_node_labels_to_integers(g, label_attribute='name')
                    node_lables = nx.get_node_attributes(converted_g, 'name')
                    grakel_g = grakel.Graph(converted_g.edges(), node_labels=node_lables)
                    previous = grakel_g
            else:
                if package == 'netrd':
                    dist = dist_metric.dist(G1=previous, G2=sub_g)
                    dist_dict[k] = dist
                if package == 'grakel':
                    # fit transform the previous graph to build model
                    gk = dist_metric(normalize=True)
                    gk.fit_transform([previous])
                    # convert the g we are iterating over to continuous
                    converted_g = nx.convert_node_labels_to_integers(g, label_attribute='name')
                    node_lables = nx.get_node_attributes(converted_g, 'name')
                    grakel_g = grakel.Graph(converted_g.edges(), node_labels=node_lables)
                    # transform with the new g, save the distance
                    result = gk.transform([grakel_g])
                    dist_dict[k] = result[0][0]
                    previous = grakel_g
    df_stats[dist_metric_name] = df_stats.index.map(dist_dict)
    print(f'{dist_metric_name} added to stats dataframe.')


# %% generate random stochastic block models
random.seed(42)
dict_rand_graphs = dict()
for i in range(50):
    sizes = [random.randrange(25, 250), random.randrange(25, 250), random.randrange(25, 250)]
    probs = [[round(random.random(), 2), 0.05, 0.02],
             [0.05, round(random.random(), 2), 0.07],
             [0.02, 0.07, round(random.random(), 2)]]
    rand_g = nx.stochastic_block_model(sizes, probs, seed=42)
    dict_rand_graphs[i] = nx.to_pandas_edgelist(rand_g, source='Source', target='Target')

df_stats_rand, common_nodes_rand = calc_network_stats(dict_rand_graphs)

# %% draw the rando graph
nx.draw_networkx(rand_g)
plt.title('Sample Random Network')
plt.show()

# %% build a list of all dates for ais months, and build edgelist dict
dates_ais = list()
for i in range(1, 12):
    start_date = f'2017-{str(i).zfill(2)}-01'
    end_date = f'2017-{str(i + 1).zfill(2)}-01'
    dates_ais.append((start_date, end_date))
dates_ais.append(('2017-12-01', '2018-01-01'))

dict_edgelist_ais_month = get_edgelists_dict([3], [8], dates_ais, loc_engine)
df_stats_ais_month, common_nodes_ais_month = calc_network_stats(dict_edgelist_ais_month)

# %% build edgelist dict for range of hyperparameters
kms = [1, 2, 3, 4, 5]
hrs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
dates = [('2017-01-01', '2018-01-01')]
dict_edgelist_ais_params = get_edgelists_dict(kms, hrs, dates, loc_engine)
df_stats_ais_params, common_nodes_ais_params = calc_network_stats(dict_edgelist_ais_params)
df_stats_ais_params['km'] = df_stats_ais_params.index.str[0]
df_stats_ais_params['hrs'] = df_stats_ais_params.index.str[5:7]

plot_network_params_comparisons(df_stats_ais_params, 'numb_nodes')

# %% compute netrd distances
netrd_dists = dict(JaccardDistance=netrd.distance.JaccardDistance(), Hamming=netrd.distance.Hamming(),
                   HammingIpsenMikhailov=netrd.distance.HammingIpsenMikhailov(), Frobenius=netrd.distance.Frobenius(),
                   PolynomialDissimilarity=netrd.distance.PolynomialDissimilarity(),
                   PortraitDivergence=netrd.distance.PortraitDivergence(),
                   OnionDivergence=netrd.distance.OnionDivergence(), QuantumSpectralJSD=netrd.distance.QuantumJSD(),
                   DegreeDivergence=netrd.distance.DegreeDivergence(),
                   ResistancePerturbation=netrd.distance.ResistancePerturbation(), NetLSD=netrd.distance.NetLSD(),
                   CommunicabilitySequence=netrd.distance.CommunicabilityJSD(),
                   IpsenMikhailov=netrd.distance.IpsenMikhailov(),
                   NonBacktrackingSpectral=netrd.distance.NonBacktrackingSpectral(),
                   NetSimile=netrd.distance.NetSimile(), DeltaCon=netrd.distance.DeltaCon())

# for the smallest graph as reference
for name, metric in netrd_dists.items():
    try:
        compute_network_distances(dict_edgelists=dict_edgelist_ais_params, df_stats=df_stats_ais_params,
                                  common_nodes=common_nodes_ais_params, dist_metric=metric, dist_metric_name=name,
                                  ref_type='previous', package='netrd')
    except Exception as e:
        print(f'{name} metric had an error:')
        print(e)

# for the previous graph as referense
for name, metric in netrd_dists.items():
    try:
        compute_network_distances(dict_edgelists=dict_edgelist_ais_month, df_stats=df_stats_ais_month,
                                  common_nodes=common_nodes_ais_month, dist_metric=metric, dist_metric_name=name,
                                  ref_type='previous', package='netrd')
    except Exception as e:
        print(f'{name} metric had an error:')
        print(e)

# %%

from grakel.kernels import ShortestPath
from grakel.kernels import RandomWalk

grakel_dists = dict(GraKeL_ShortestPath=ShortestPath, GraKel_RandomWalk=RandomWalk)

for name, metric in grakel_dists.items():
    try:
        compute_network_distances(dict_edgelists=dict_edgelist_ais_params, df_stats=df_stats_ais_params,
                                  common_nodes=common_nodes_ais_params, dist_metric=metric, dist_metric_name=name,
                                  ref_type='smallest', package='grakel')
    except Exception as e:
        print(f'{name} metric had an error:')
        print(e)

for name, metric in grakel_dists.items():
    try:
        compute_network_distances(dict_edgelists=dict_edgelist_ais_month, df_stats=df_stats_ais_month,
                                  common_nodes=common_nodes_ais_month, dist_metric=metric, dist_metric_name=name,
                                  ref_type='previous', package='grakel')
    except Exception as e:
        print(f'{name} metric had an error:')
        print(e)

#
# #%%
# df_1km_2h = gsta.get_edgelist(edge_table='cargo_edgelist_1km', engine=loc_engine, loiter_time=2)
# g_1km_2h = nx.from_edgelist(df_1km_2h[['Source_id', 'Target_id']].values)
# g_1km_2h = nx.convert_node_labels_to_integers(g_1km_2h, label_attribute='name')
# nodes = nx.get_node_attributes(g_1km_2h, 'name')
# grakel_1km_2h = grakel.Graph(g_1km_2h.edges(), node_labels=nodes)
#
#
#
# # %%
# gk = ShortestPath(normalize=True, with_labels=True)
# gk.fit_transform([grakel_1km_2h])
#
# dist_dict = dict()
# for k, v in df_dict.items():
#     # convert from df to graph
#     g = nx.from_edgelist(v[['Source', 'Target']].values)
#     g = nx.convert_node_labels_to_integers(g, label_attribute='name')
#     node_lables = nx.get_node_attributes(g, 'name')
#     grakel_g = grakel.Graph(g.edges(), node_labels=node_lables)
#     # compute distance between two graphs using the grakel package
#     result = gk.transform([grakel_g])
#     dist_dict[k] = result[0][0]
#
# df_stats['grakel'] = df_stats.index.map(dist_dict)
# plot_network_params_comparisons(df_stats, 'grakel')
# # %% for randos
#
# import grakel
# from grakel.kernels import ShortestPath
#
# smallest_grakel = grakel.Graph(smallest_g.edges())
#
# gk = ShortestPath(normalize=True, with_labels=False)
# gk.fit_transform([smallest_grakel])
#
# dist_dict = dict()
# for k, v in df_dict.items():
#     # convert from df to graph
#     g = nx.from_edgelist(v[['Source', 'Target']].values)
#     grakel_g = grakel.Graph(g.edges())
#     # compute distance between two graphs using the grakel package
#     result = gk.transform([grakel_g])
#     dist_dict[k] = result[0][0]
#
# df_stats['grakel'] = df_stats.index.map(dist_dict)
#
# # %%
# df_stats['grakel'].plot()
# plt.xticks(rotation=60)
# plt.title('GraKel Shortest Path Kernel for AIS Date')
# plt.show()
# # %%
# df_stats = df_stats_ais_month
#
# df_stats[['JaccardDistance',
#           'Hamming', 'HammingIpsenMikhailov',
#           'PolynomialDissimilarity', 'PortraitDivergence', 'QuantumSpectralJSD',
#           'DegreeDivergence',
#           'IpsenMikhailov']].plot()
# plt.xticks(rotation=60)
# plt.legend(bbox_to_anchor=(1, 1))
# plt.title('Small-Scale NetRD Metrics for Random Graphs')
# plt.show()
#
# df_stats[['Frobenius', 'DeltaCon']].plot()
# plt.xticks(rotation=60)
# plt.title('Large-Scale NetRD Metrics for Random Graphs')
# plt.show()
#
# df_stats[['numb_nodes', 'numb_edges']].plot()
# plt.title('Edge and Node Size for Random Graphs')
# plt.xticks(rotation=60)
# plt.show()
#
# df_stats[['avg_degree', 'avg_betweenness', 'avg_centrality',
#           'avg_clust_coeff', 'transitivity']].plot()
# plt.xticks(rotation=60)
# plt.title('Network Metrics for Random Graphs')
# plt.show()
