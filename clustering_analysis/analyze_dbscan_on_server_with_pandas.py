import datetime
import pandas as pd
from gnact import clust, network, utils
import db_config

import warnings  # postgres datatype geom not recognized

warnings.filterwarnings('ignore')

# create the engine to the database
engine = utils.connect_engine(db_config.colone_cargo_params, print_verbose=True)
source_table = 'uid_positions_jan'
results_table = 'clustering_results_hdbscan'
end_date = '2017-02-01'  # just select the first month of January for testing

# %% Create needed accessory tables and ensure they are clean.  also get uid list
sql_get_cols = f"""
 SELECT column_name
  FROM information_schema.columns
 WHERE table_schema = 'public'
   AND table_name   = '{results_table}'
   AND column_name != 'id';"""

clust_results_df = pd.read_sql(sql_get_cols, engine)

# %% calculate stops
# use the get_edgelist function designed for generating networks as an application of static trip segmentation.
# we will use this as our ground truth with the same parameters for time and space as our demo.
# cargo_edgelinst_xkm is a table created by rolling up clusters and calculating the nearest site to each point.
# it takes hours to compute and resides on the server.  See GNACT documentation for more information.
df_stops = network.get_edgelist('cargo_edgelist_5km', engine=engine, loiter_time=6)
# we will need site_id for joining with the rest of the data
df_stops['site_id'] = df_stops['Source_id']
# filter down to stops within the time period of interest
df_stops = df_stops[df_stops['source_depart'] < end_date]

# uncomment this for test run with 636016432 UID
# df_stops = df_stops[df_stops['uid'] == '636016432']

# %%
results_dict = dict()
# iterate through each column name, select the relevant info from the db, and calculate the stats using the known stops.
for col in clust_results_df['column_name']:
    first_tick = datetime.datetime.now()
    # this sql query will be used to select each column with clustering results already computed across the entire dataset.
    # this could be millions of points with their clust_id, so we cant hold as a df.
    # this SQL query is analagous to
    # 1) running the gnact.clust.calc_centers() function
    # 2) merging the reulst with a rollup of total number of points in a cluster
    # 3) running the gnact.clust.calc_nn() function with the resulting df and the original df_sites
    read_sql = f"""    
    -- creates summary for the clustering results joined with original position info
        with summary as (
            select c.{col} as clust_result, pos.uid as uid, count(pos.id) as total_points, 
            ST_Centroid(ST_union(pos.geom)) as avg_geom
            from {source_table} as pos, {results_table} as c
            where c.id = pos.id
            and c.{col} is not null
            --and pos.uid = '636016432' --uncomment this for test run
            group by clust_result, uid)
    --from the summary, concats cluster id and uid, gets distance, and cross joins 
    --to get the closest site for each cluster
    select concat(summary.clust_result::text, '_', summary.uid::text) as clust_id, 
        summary.total_points,
        sites.site_id as nearest_site_id, 
        sites.port_name as site_name,
        (ST_Distance(sites.geom::geography, summary.avg_geom::geography)/1000) AS nearest_site_dist_km
        from summary
    cross join lateral --gets only the nearest port
        (select sites.site_id, sites.port_name, sites.geom
        from sites
        order by sites.geom <-> avg_geom limit 1)
        as sites"""

    # run the sql query with the current column
    df_nearest_sites = pd.read_sql(read_sql, engine)

    # correct clusters are within distance threshold of a visited point so we filter by dist_threshold
    df_clust_rollup_correct = df_nearest_sites[(df_nearest_sites.nearest_site_dist_km < 5)]

    # now group stops and clust_rollup by their site_ids, and select just one column.
    # the result is a series with site_id as index that can be used to calc precision, recall, and f1 measure
    df_stops_grouped = df_stops.groupby('site_id').agg('count').iloc[:, 0]
    df_rollup_grouped = df_clust_rollup_correct.groupby(['nearest_site_id']).agg('count').iloc[:, 0]

    # get the proportion of each site within stops to use for the recall
    total_prop_stops = df_stops_grouped / df_stops_grouped.sum()
    # get raw recall, which we calc by cluster.  therefore recall is number of clusters found at a site
    # divided by the total number of clusters at that site.  If the value is more than 1, set it to 1.
    recall_raw = (df_rollup_grouped / df_stops_grouped)
    recall_raw[recall_raw > 1] = 1
    # now multiply raw_recall by the total proportion to get a weighted value, and sum it.
    recall = (recall_raw * total_prop_stops).sum()
    # precision is the proportion of correct clusters to all clusters found.  since we are using df_stops,
    # correct clusters are any clusters with a calculated distance less than the distance threshold.
    try:
        precision = len(df_clust_rollup_correct) / len(df_nearest_sites)
    except Exception as e:
        precision = 0
    # now determine f1 measure
    f_measure = 2 * ((precision * recall) / (precision + recall))

    # build the stats_dict
    stats_dict = {'precision': round(precision, 4),
                  'recall': round(recall, 4),
                  'f1': round(f_measure, 4),
                  'average_points': round(df_nearest_sites['total_points'].mean(), 4),
                  'total_clusters': len(df_nearest_sites['total_points']),
                  'average_nearest_site': round((df_nearest_sites['nearest_site_dist_km'].mean()), 4)}
    last_tock = datetime.datetime.now()
    lapse = last_tock - first_tick
    print(f'Processing Done for {col}.  Total time elapsed: {lapse}.')
    print(stats_dict)
    results_dict[col] = stats_dict

# %%
results_df = pd.DataFrame(results_dict).T
# uncomment for test with sample uid
# results_df.to_csv(f'stats_{results_table}_636016432.csv')
results_df.to_csv(f'stats_{results_table}.csv')
