import datetime
from multiprocessing import Pool

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from multiprocessing import Pool
from itertools import repeat

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

from importlib import reload

reload(gsta)

rollup_table = 'clustering_rollup'
results_table = 'clustering_results'

# %% Create needed accessory tables and ensure they are clean.  also get uid list
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
c = conn.cursor()
c.execute(f"""CREATE TABLE IF NOT EXISTS {rollup_table}
(   name text,
    total_clusters int, 
    avg_points float,
    average_dist_nearest_port float,
    total_sites int,
    common_sites int,
    all_sites int,
    site_names text[],
    site_ids int[],
    precision float,
    recall float,
    f1 float
);""")
conn.commit()
c.close()

# get all the columns from the clustering_results
# need to add in an option to pull the names in the rollup table, find the intersection,
# and only analyze new tables.
c = conn.cursor()
c.execute(f"""
 SELECT column_name
  FROM information_schema.columns
 WHERE table_schema = 'public'
   AND table_name   = '{results_table}'
   AND column_name != 'id';""")
clust_results_cols = c.fetchall()
c.close()


c = conn.cursor()
c.execute(f"""SELECT DISTINCT (name) from {rollup_table};""")
completed_cols = c.fetchall()
c.close()
conn.close()

to_do_cols = np.setdiff1d(clust_results_cols,completed_cols)


#%%
def get_cluster_rollup(col):
    print(f'Starting {col}...')
    sql_analyze = f"""
    --return clust_id, total points, avg_geom, nearest_site_id, site name and site geom
    with final as (
    -- creates summary for the clustering results joined with original position info
        with summary as (
            select c.{col} as clust_result, pos.uid as uid, count(pos.id) as total_points, 
            ST_Centroid(ST_union(pos.geom)) as avg_geom
            from uid_positions as pos, {results_table} as c
            where c.id = pos.id
            and c.{col} is not null
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
        as sites
        )
    --aggregates all data for this set of results into one row
    insert into {rollup_table} (name, total_clusters, avg_points, average_dist_nearest_port,
                          total_sites, site_names, site_ids)
        select '{col}',
        count(final.clust_id), 
        avg(final.total_points), 
        avg(final.nearest_site_dist_km),
        count(distinct(final.site_name)),
        array_agg(distinct(final.site_name)),
        array_agg(distinct(final.nearest_site_id))
    from final
    """
    # run the sql_analyze
    conn_pooled = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    c_pooled = conn_pooled.cursor()
    c_pooled.execute(sql_analyze)
    conn_pooled.commit()
    c_pooled.close()
    conn_pooled.close()
    print(f'Completed {col}')

#%%
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

# execute the function with pooled workers

with Pool(5) as p:
    p.map(get_cluster_rollup, to_do_cols)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)
conn.close()
#%%
# once the rollup table is populated with the total_clusters, total_sites, site_names, site_ids,
# this query will determine the precision, recall, and f1 score.  This can be once against the entire table
# so it does not to be parallelized like the get_cluster_rollup function that must be run for each column.
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
c = conn.cursor()
c.execute(f"""
-- the metrics table will be where all the final values (except f1) will be stored and used to update the rollup
with metrics as (
    --the p_r temp table will be used to determine the precision and recall
	with p_r as (
        --count the total_sites, common_sites, and all_sites column
        with ports_3km as (
        --select all the nearest port_ids less than the set distance 
        --that have a minimum number of points.
        select ARRAY(select distinct (n.nearest_site_id)
        from nearest_site_jan as n
        where n.nearest_site_dist_km < 3
        group by nearest_site_id
        -- filters out ports with x or fewer points, which are not likely site visits
        having count(n.nearest_site_id) > 10) as all_sites)
            -- get the intersection of all sites and the clustering sites as common_sites,
            -- all of the sites with min points within 3km of a port as all sites
            select c.name, c.total_sites,
            cardinality(ports_3km.all_sites & c.site_ids) as common_sites,
            cardinality(ports_3km.all_sites) as all_sites
            from ports_3km, {rollup_table} as c)
	--determine the precision and recall
	select p_r.name, p_r.total_sites, p_r.common_sites, p_r.all_sites,
	p_r.common_sites::float / p_r.total_sites::float as precision,
	p_r.common_sites::float / p_r.all_sites::float as recall
	from p_r)
--update the clustering results and calculate the f1 measure
update {rollup_table} set
common_sites = metrics.common_sites, 
all_sites = metrics.all_sites, 
precision = metrics.precision, 
recall = metrics.recall, 
f1 = (2*metrics.precision*metrics.recall) /(metrics.precision+metrics.recall)
from metrics
where {rollup_table}.name = metrics.name
;""")
conn.commit()
c.close()