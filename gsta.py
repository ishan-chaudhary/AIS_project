#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:44:23 2020

@author: patrickmaus
"""
# time tracking
import datetime
import os

import psycopg2
from sqlalchemy import create_engine

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances


# %% Database connection functions
def connect_psycopg2(params):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
        c = conn.cursor()
        # Execute a statement
        print('PostgreSQL database version:')
        c.execute('SELECT version()')
        db_version = c.fetchone()
        print(db_version)
        # close the communication with the PostgreSQL
        c.close()
        print('Connection created for', params['database'])
        return conn

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)


def connect_engine(params):
    print('Creating Engine...')
    try:
        engine = create_engine('postgresql://{}@{}:{}/{}'.format(params['user'],
                                                                 params['host'],
                                                                 params['port'],
                                                                 params['database']))
        print('Engine created for', params['database'])
        return engine

    except:
        print('Engine Creation failed.')


# %% Database management functions
def drop_table(table, conn):
    c = conn.cursor()
    c.execute('drop table if exists {} cascade'.format(table))
    conn.commit()
    c.close()


def dedupe_table(table, conn):
    c = conn.cursor()
    c.execute("""CREATE TABLE tmp as
          (SELECT * from (SELECT DISTINCT * FROM {}) as t);""".format(table))
    c.execute("""DELETE from {};""".format(table))
    c.execute("""INSERT INTO {} SELECT * from tmp;""".format(table))
    c.execute("""DROP TABLE tmp;""")
    conn.commit()
    c.close()


# %% DBSCAN run helper functions
def sklearn_dbscan(source_table, new_table_name, eps, min_samples,
                   mmsi_list, conn, engine, from_schema_name, to_schema_name,
                   lat='lat', lon='lon'):
    for mmsi in mmsi_list:
        # next get the data for the mmsi
        read_sql = """SELECT id, mmsi, {0}, {1}
                    FROM {2}.{3}
                    WHERE mmsi = '{4}'
                    ORDER by time""".format(lat, lon, from_schema_name, source_table, mmsi[0])
        df = pd.read_sql_query(read_sql, con=engine)

        # format data for dbscan
        X = (np.radians(df.loc[:, ['lon', 'lat']].values))
        x_id = df.loc[:, 'id'].values

        # execute sklearn's DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree',
                        metric='haversine', n_jobs=-1)
        dbscan.fit(X)

        # gather the output as a dataframe
        results_dict = {'id': x_id, 'lat': np.degrees(X[:, 1]),
                        'lon': np.degrees(X[:, 0]), 'clust_id': dbscan.labels_}
        df_results = pd.DataFrame(results_dict)
        # drop all -1 clust_id, which are all points not in clusters
        df_results = df_results[df_results['clust_id'] != -1]
        df_results['mmsi'] = mmsi[0]

        # write df to databse
        df_results.to_sql(name=new_table_name, con=engine, schema=to_schema_name,
                          if_exists='append', method='multi', index=False)

        print('DBSCAN complete for MMSI {}.'.format(mmsi[0]))


def sklearn_dbscan_rollup(source_table, new_table_name, eps, min_samples,
                          conn, engine, from_schema_name, to_schema_name,
                          lat='average_lat', lon='average_lon'):
    read_sql = """SELECT clust_id, {0}, {1}
                FROM {2}.{3}""".format(lat, lon, from_schema_name, source_table)
    df = pd.read_sql_query(read_sql, con=engine)

    # format data for dbscan.  note lon/lat order
    X = (np.radians(df.loc[:, [lon, lat]].values))
    x_id = df.loc[:, 'clust_id'].values

    # execute sklearn's DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree',
                    metric='haversine', n_jobs=-1)
    dbscan.fit(X)

    # gather the output as a dataframe
    results_dict = {'id': x_id, 'lat': np.degrees(X[:, 1]),
                    'lon': np.degrees(X[:, 0]), 'super_clust_id': dbscan.labels_}
    df_results = pd.DataFrame(results_dict)
    # drop all -1 clust_id, which are all points not in clusters
    df_results = df_results[df_results['super_clust_id'] != -1]

    # write df to databse
    df_results.to_sql(name=new_table_name, con=engine, schema=to_schema_name,
                      if_exists='replace', method='multi', index=False)

    print('DBSCAN complete for {}.'.format(source_table))


def postgres_dbscan(source_table, new_table_name, eps, min_samples,
                    mmsi_list, conn, from_schema_name, to_schema_name):
    # drop table if it exists
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS {}.{}""".format(to_schema_name, new_table_name))
    conn.commit()
    c.close()
    # create the new table in the new schema to hold the results for each
    # point in the source table, which will include the cluster id.
    # Postgres or sklearn, we will only write points that have a valid clust_id
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS {}.{}
    (   id integer,
        mmsi text,
        lat numeric,
        lon numeric,
        clust_id int
    );""".format(to_schema_name, new_table_name))
    conn.commit()
    # iterate through each mmsi and insert into the new schema and table
    # the id, mmsi, lat, lon, and cluster id using the epsilon in degrees.
    # Only write back when a position's cluster id is not null.
    for mmsi in mmsi_list:
        dbscan_postgres_sql = """INSERT INTO {0}.{1} (id, mmsi, lat, lon, clust_id)
        WITH dbscan as (
        SELECT id, mmsi, lat, lon,
        ST_ClusterDBSCAN(geom, eps := {2}, minpoints := {3})
        over () as clust_id
        FROM {6}.{4}
        WHERE mmsi = '{5}')
        SELECT * from dbscan
        WHERE clust_id IS NOT NULL;""".format(schema_name, new_table_name, str(eps),
                                              str(min_samples), source_table, mmsi[0], from_schema_name)
        # execute dbscan script
        c = conn.cursor()
        c.execute(dbscan_postgres_sql)
        conn.commit()
        c.close()
        print('MMSI {} complete.'.format(mmsi[0]))
    print('DBSCAN complete, {} created'.format(new_table_name))


def make_tables_geom(table, schema_name, conn):
    """A function to make the provided table in provided schema a geometry.
    Table needs to have an existing lat and lon column and not already have a geom column."""
    # add a geom column to the new table and populate it from the lat and lon columns
    c = conn.cursor()
    c.execute("""ALTER TABLE {}.{} ADD COLUMN
                geom geometry(Point, 4326);""".format(schema_name, table))
    conn.commit()
    c.execute("""UPDATE {}.{} SET
                geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);""".format(schema_name, table))
    conn.commit()
    c.close()


def get_mmsi_list(source_table, conn):
    """For a given source table, return the list of distinct MMSIs."""
    c = conn.cursor()
    c.execute("""SELECT DISTINCT(mmsi) FROM {};""".format(source_table))
    mmsi_list = c.fetchall()
    print('{} total MMSIs returned from {}'.format(str(len(mmsi_list)), source_table))
    c.close()
    return mmsi_list


def create_schema(schema_name, conn, drop_schema=True, with_date=True):
    """Create the given schamea name, with the option of dropping any
    exisiting schema with the same name.  Can append date to schema."""
    # add the date the run started if desired
    if with_date == True:
        date = str(datetime.date.today()).replace('-', '_')
        schema_name = schema_name + '_' + date

    # if desired, drop existing schema name
    if drop_schema == True:
        c = conn.cursor()
        c.execute("""DROP SCHEMA IF EXISTS {} CASCADE;""".format(schema_name))
        conn.commit()
        print('Old version of schema {} deleted if exists'.format(schema_name))

    # make a new schema to hold the results
    c = conn.cursor()
    c.execute("""CREATE SCHEMA IF NOT EXISTS {};""".format(schema_name))
    conn.commit()
    print('New schema {} created if it did not exist.'.format(schema_name))
    return schema_name


def execute_dbscan(source_table, from_schema_name, to_schema_name, eps_samples_params,
                   conn, engine, method='sklearn', drop_schema=False):
    # check to make sure the method type is correct
    method_types = ['sklearn_mmsi', 'postgres_mmsi', 'sklearn_rollup']
    if method not in method_types:
        print("Argument 'method' must be 'sklearn_mmsi', 'sklearn_rollup', or 'postgres_mmsi'.")
        return

    print('{} DBSCAN begun at:'.format(method), datetime.datetime.now())
    outer_tick = datetime.datetime.now()

    if method in ['sklearn_mmsi', 'postgres_mmsi']:
        # get the mmsi list from the source table.
        mmsi_list = get_mmsi_list(source_table, conn)
    else:
        pass

    # itearate through the epsilons and samples given
    for p in eps_samples_params:
        eps_km, min_samples = p
        inner_tick = datetime.datetime.now()

        print("""Starting processing on {} DBSCAN with
        eps_km={} and min_samples={} """.format(method, str(eps_km), str(min_samples)))

        # this formulation will yield epsilon based on km desired.
        # DBSCAN in postgres only works with geom, so distance is based on
        # cartesian plan distance calculations.  This is only approximate
        # because the length of degrees are different for different latitudes.
        # however it should be fine for small distances.
        kms_per_radian = 6371.0088
        eps = eps_km / kms_per_radian

        # make the new table name
        new_table_name = (method + '_' + str(eps_km).replace('.', '_') +
                          '_' + str(min_samples))

        if method == 'postgres_mmsi':
            postgres_dbscan(source_table, new_table_name, eps, min_samples,
                            mmsi_list, conn, from_schema_name, to_schema_name, )
        elif method == 'sklearn_mmsi':
            sklearn_dbscan(source_table, new_table_name, eps, min_samples,
                           mmsi_list, conn, engine, from_schema_name, to_schema_name, )
        elif method == 'sklearn_rollup':
            sklearn_dbscan_rollup(source_table, new_table_name, eps, min_samples,
                                  conn, engine, from_schema_name, to_schema_name, )

        # add geom colum to the new tables
        make_tables_geom(new_table_name, to_schema_name, conn)

        # timekeeping for each iteration
        tock = datetime.datetime.now()
        lapse = tock - inner_tick
        print('Time elapsed for this iteration: {}'.format(lapse))

    # timekeeping for entire approach
    tock = datetime.datetime.now()
    lapse = tock - outer_tick
    print('Time elapsed for entire process: {}'.format(lapse))
    print('Results for {} method stored in {} schema.'.format(method, to_schema_name))


## Analyze DBSCAN helper functions
def get_ports_wpi(engine):
    """Creates a df with all the ports from the WPI"""
    ports = pd.read_sql_table(table_name='wpi', con=engine, columns=['index_no', 'port_name', 'latitude', 'longitude'])
    ports = ports.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'index_no': 'port_id'})
    return ports


def get_ports_labeled(table_name, engine):
    """Creates a df with all the labeled ports derived from earlier process in the data ETL"""
    ports_labeled = pd.read_sql_table(table_name, con=engine,
                                      columns=['port_name', 'nearest_port_id', 'count'])
    return ports_labeled


def calc_dist(df_results, clust_id_value, engine):
    """This function finds the center of a cluster from dbscan results,
    determines the nearest port, and finds the average distance for each
    cluster point from its cluster center.  Returns a df."""

    ports_wpi = get_ports_wpi(engine)

    # make a new df from the df_results grouped by cluster id
    # with the mean for lat and long
    df_centers = (df_results[[clust_id_value, 'lat', 'lon']]
                  .groupby(clust_id_value)
                  .mean()
                  .rename({'lat': 'average_lat', 'lon': 'average_lon'}, axis=1)
                  .reset_index())

    # Now we are going to use sklearn's KDTree to find the nearest neighbor of
    # each center for the nearest port.
    points_of_int = np.radians(df_centers.loc[:, ['average_lat', 'average_lon']].values)
    candidates = np.radians(ports_wpi.loc[:, ['lat', 'lon']].values)
    tree = BallTree(candidates, leaf_size=30, metric='haversine')

    nearest_list = []
    for i in range(len((points_of_int))):
        dist, ind = tree.query(points_of_int[i, :].reshape(1, -1), k=1)
        nearest_dict = {clust_id_value: df_centers.iloc[i].loc[clust_id_value],
                        'nearest_port_id': ports_wpi.iloc[ind[0][0]].loc['port_id'],
                        'nearest_port_dist': dist[0][0] * 6371.0088}
        nearest_list.append(nearest_dict)
    df_nearest = pd.DataFrame(nearest_list)
    df_centers = pd.merge(df_centers, df_nearest, how='left', on=clust_id_value)

    # find the average distance from the centerpoint
    # We'll calculate this by finding all of the distances between each point in
    # df_results and the center of the cluster.  We'll then take the min and the mean.
    haver_list = []
    for i in df_centers[clust_id_value]:
        X = (np.radians(df_results[df_results[clust_id_value] == i]
                        .loc[:, ['lat', 'lon']].values))
        Y = (np.radians(df_centers[df_centers[clust_id_value] == i]
                        .loc[:, ['average_lat', 'average_lon']].values))
        haver_result = (haversine_distances(X, Y)) * 6371.0088  # km to radians
        haver_dict = {clust_id_value: i, 'min_dist_from_center': haver_result.min(),
                      'max_dist_from_center': haver_result.max(),
                      'average_dist_from_center': np.mean(haver_result)}
        haver_list.append(haver_dict)

    # merge the haver results back to df_centers
    haver_df = pd.DataFrame(haver_list)
    df_centers = pd.merge(df_centers, haver_df, how='left', on=clust_id_value)

    # create "total cluster count" column through groupby
    clust_size = (df_results[['lat', clust_id_value]]
                  .groupby(clust_id_value)
                  .count()
                  .reset_index()
                  .rename({'lat': 'total_clust_count'}, axis=1))
    # merge results back to df_Centers
    df_centers = pd.merge(df_centers, clust_size, how='left', on=clust_id_value)
    return df_centers


def calc_harmonic_mean(precision, recall):
    return (2 * ((precision * recall) / (precision + recall)))


def calc_stats(df_rollup, ports_labeled, engine, noise_filter):
    df_ports_labeled = get_ports_labeled(ports_labeled, engine)
    # determine the recall, precision, and f-measure
    # drop all duplicates in the rollup df to get just the unique port_ids
    # join to the list of all ports within a set distance of positions.
    # the > count allows to filter out noise where only a handful of positions
    # are near a given port.  Increasing this will increase recall because there
    # are fewer "hard" ports to indetify with very little activity.
    df_stats = pd.merge((df_rollup[df_rollup['nearest_port_dist'] < 5]
                         .drop_duplicates('nearest_port_id')),
                        df_ports_labeled[df_ports_labeled['count'] > noise_filter],
                        how='outer', on='nearest_port_id', indicator=True)
    # this df lists where the counts in the merge.
    # left_only are ports only in the dbscan.  (false positives for dbscan)
    # right_only are ports only in the ports near positions.  (false negatives for dbscan)
    # both are ports in both datasets.  (true positives for dbscan)
    values = (df_stats['_merge'].value_counts())
    # recall is the proporation of relevant items selected
    # it is the number of true positives divided by TP + FN
    stats_recall = (values['both'] /
                    (len((df_ports_labeled[df_ports_labeled['count'] > noise_filter])
                         .drop_duplicates('nearest_port_id'))))
    # precision is the proportion of selected items that are relevant.
    # it is the number of true positives our of all items selected by dbscan.
    stats_precision = values['both'] / len(df_rollup.drop_duplicates('nearest_port_id'))
    # now find the f_measure, which is the harmonic mean of precision and recall
    stats_f_measure = calc_harmonic_mean(stats_precision, stats_recall)
    return stats_f_measure, stats_precision, stats_recall


def df_to_table_with_geom(df, rollup_table_name, schema_name, conn, engine):
    # make a new table with the df
    df.to_sql(name=rollup_table_name, con=engine, schema=schema_name,
              if_exists='replace', method='multi', index=False)
    # add a geom column to the new table and populate it from the lat and lon columns
    c = conn.cursor()
    c.execute("""ALTER TABLE {}.{} ADD COLUMN
                geom geometry(Point, 4326);""".format(schema_name, rollup_table_name))
    conn.commit()
    c.execute("""UPDATE {}.{} SET
                geom = ST_SetSRID(ST_MakePoint(average_lon, average_lat), 4326);""".format(schema_name,
                                                                                           rollup_table_name))
    conn.commit()
    c.close()


def analyze_dbscan(method_used, conn, engine, schema_name, ports_labeled,
                   eps_samples_params, id_value, clust_id_value, noise_filter):
    rollup_list = []
    path = '/Users/patrickmaus/Documents/projects/AIS_project/DBSCAN/rollups/{}/'.format(schema_name)
    if not os.path.exists(path): os.makedirs(path)
    # timekeeping
    outer_tick = datetime.datetime.now()

    # itearate through the epsilons and samples given
    for p in eps_samples_params:
        eps_km, min_samples = p
        print("""Starting analyzing DBSCAN results with eps_km={} and min_samples={}""".format(str(eps_km),
                                                                                               str(min_samples)))
        # timekeeping
        tick = datetime.datetime.now()
        # make table name, and pull the results from the correct sql table.
        table = (method_used + '_' + str(eps_km).replace('.', '_') + '_' + str(min_samples))
        df_results = pd.read_sql_table(table_name=table, con=engine, schema=schema_name,
                                       columns=[id_value, 'lat', 'lon', clust_id_value])
        # since we created clusters by mmsi, we are going to need to redefine
        # clust_id to include the mmsi and clust_id
        # since we created clusters by mmsi, we are going to need to redefine
        # clust_id to include the mmsi and clust_id
        df_results['clust_id'] = (df_results[id_value] + '_' +
                                  df_results[clust_id_value].astype(int).astype(str))

        # determine the cluster center point, and find the distance to nearest port
        print('Starting distance calculations... ')
        try:
            df_rollup = calc_dist(df_results, clust_id_value, engine)
            print('Finished distance calculations. ')
        except:
            print('There were no clusters for this round.  Breaking loop...')
            continue
        # calculate stats
        print('Starting stats calculations...')
        stats_f_measure, stats_precision, stats_recall = calc_stats(df_rollup, ports_labeled, engine,
                                                                    noise_filter=noise_filter)
        print("finished stats calculations.")
        # roll up written to csv and to table
        df_rollup['eps_km'] = eps_km
        df_rollup['min_samples'] = min_samples
        rollup_name = ('summary_' + str(eps_km).replace('.', '_') +
                       '_' + str(min_samples))
        df_rollup.to_csv(path + rollup_name + '.csv')
        df_to_table_with_geom(df_rollup, rollup_name, schema_name, conn, engine)

        # the rollup dict contains multiple different metrics options.
        rollup_dict = {'eps_km': eps_km, 'min_samples': min_samples,
                       'params': (str(eps_km) + '_' + str(min_samples)),
                       # number of clusters in the run
                       'numb_clusters': len(np.unique(df_rollup[clust_id_value])),
                       # average positions per each cluster in each run
                       'average_cluster_count': np.mean(df_rollup['total_clust_count']),
                       # distance metrics
                       'average_nearest_port_from_center': np.mean(df_rollup['nearest_port_dist']),
                       'average_dist_from_center': np.mean(df_rollup['average_dist_from_center']),
                       'average_max_dist_from_center': np.mean(df_rollup['max_dist_from_center']),
                       # stats
                       'f_measure': stats_f_measure,
                       'precision': stats_precision,
                       'recall': stats_recall}
        rollup_list.append(rollup_dict)

        # timekeeping
        tock = datetime.datetime.now()
        lapse = tock - tick
        print('All processing for this run complete.')
        print('Time elapsed: {}'.format(lapse))

    # Make the final_df from the rollup and save to csv.
    final_df = pd.DataFrame(rollup_list).round(3)
    final_df['params'] = (final_df['eps_km'].astype('str') + '_'
                          + final_df['min_samples'].astype('str'))
    final_df.set_index('params', inplace=True)
    final_df.to_csv(path + 'final_summary_' + method_used + '.csv')

    # timekeeping
    tock = datetime.datetime.now()
    lapse = tock - outer_tick
    print('All processing for complete.  Data written to schema {}'.format(schema_name))
    print('Time elapsed: {}'.format(lapse))
    return final_df


# Network Analysis
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


def get_weighted_edgelist(df_edgelist):
    # This produces a df that is the summarized edge list with weights
    # for the numbers of a time a ship goes from the source node to the target node.
    # The code executes groupby the source/target id/name, count all the rows, drop the time fields,
    # rename the remaining column from mmsi to weight, and reset the index
    df_edgelist_weighted = (df_edgelist.groupby(['Source_id', 'Source',
                                                 'Target_id', 'Target'])
                            .count()
                            .drop(['source_depart', 'target_arrival'], axis=1)
                            .rename(columns={'mmsi': 'weight'})
                            .reset_index())
    return df_edgelist_weighted



def plot_mmsi(mmsi, df_edgelist):
    # this function will plot the path of a given mmsi across an edgelist df.
    mmsi_edgelist = df_edgelist[df_edgelist['mmsi'] == mmsi].reset_index(drop=True)
    mmsi_edgelist = mmsi_edgelist[['Source', 'source_depart', 'Target', 'target_arrival']]
    # build the graph
    G = nx.from_pandas_edgelist(mmsi_edgelist, source='Source', target='Target',
                                edge_attr=True, create_using=nx.MultiDiGraph)
    # get positions for all nodes using circular layout
    pos = nx.circular_layout(G)
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


def plot_from_source(source, df):
    # this function will plot all the nodes visited from a given node
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
    nx.draw_networkx_edges(G, pos, alpha=0.5, width=scale_range(weights, .5, 5))
    # plot the title and turn off the axis
    plt.title('Weighted Network Plot for {} Port as Source'.format(source.title()),
              fontsize=16)
    plt.axis('off')
    # # make a test boc for the weights.  Can be plotted on edge, but crowded
    # box_str = 'Weights out from {}: \n'.format(source.title())
    # for neighbor, values in G[source.upper()].items():
    #     box_str = (box_str + neighbor.title() + ' - ' +
    #                str(values[0]['weight']) + '\n')

    plt.text(-1.2, -1.2, 'Edge Weights are scaled between 0.5 and 5 for visualization.', fontsize=12,
          verticalalignment='top', horizontalalignment='left')
    plt.show()  # display

    print(df_g[['Target','weight']].sort_values('weight', ascending=False).reset_index())


def scale_range(input, min, max):
    # scale an input array-like to a mininum and maximum number
    # the input array must be of a floating point array
    # it will mutate in place
    # min and max can be integers
    input = np.array(input).astype('float')
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input
