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


# %% Create needed accessory tables and ensure they are clean.  also get uid list
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS clustering_rollup
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


#%% get all the columns from the clustering_results
# need to add in an option to pull the names in the rollup table, find the intersection,
# and only analyze new tables.
c = conn.cursor()
c.execute("""
 SELECT column_name
  FROM information_schema.columns
 WHERE table_schema = 'public'
   AND table_name   = 'clustering_results'
   AND column_name != 'id';""")
clust_results_cols = c.fetchall()
c.close()
conn.close()


#%%
def get_cluster_rollup(col):
    print(f'Starting {col[0]}...')
    sql_analyze = f"""
    --return clust_id, total points, avg_geom, nearest_port_id, site name and site geom
    with final as(
    -- creates summary for the clustering results joined with original poistion info
    with summary as (
        select c.{col[0]} as clust_result, pos.uid as uid, count(pos.id) as total_points, 
        ST_Centroid(ST_union(pos.geom)) as avg_geom
        from uid_positions as pos, clustering_results as c
        where c.id = pos.id
        and c.{col[0]} is not null
        group by clust_result, uid
    )
    --from the summary, concats cluster id and uid, gets distance, and cross joins 
    --to get the closest site for each cluster
    select concat(summary.clust_result::text, '_', summary.uid::text) as clust_id, 
        summary.total_points,
        sites.site_id as nearest_port_id, 
        sites.port_name as site_name,
        (ST_Distance(sites.geom::geography, summary.avg_geom::geography)/1000) AS nearest_port_dist_km
        from summary
    cross join lateral
        (select sites.site_id, sites.port_name, sites.geom
        from sites
        order by sites.geom <-> avg_geom limit 1)
        as sites
        )
    --aggregates all data for this set of results into one row
    insert into clustering_rollup (name, total_clusters, avg_points, average_dist_nearest_port,
                          total_sites, site_names, site_ids)
        select '{col[0]}',
        count(final.clust_id), 
        avg(final.total_points), 
        avg(final.nearest_port_dist_km),
        count(distinct(final.site_name)),
        array_agg(distinct(final.site_name)),
        array_agg(distinct(final.nearest_port_id))
    from final
    """
    # run the sql_analyze
    conn_pooled = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
    c_pooled = conn_pooled.cursor()
    c_pooled.execute(sql_analyze)
    conn_pooled.commit()
    c_pooled.close()
    conn_pooled.close()
    print(f'Completed {col[0]}')

#%%
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

# execute the function with pooled workers

with Pool(38) as p:
    p.map(get_cluster_rollup, clust_results_cols)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)
conn.close()
#%%
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params, print_verbose=False)
c = conn.cursor()
c.execute("""--find all ports within 5km of each point
with metrics as (
	with p_r as (
		with ports_3km as (
		--select all the nearest port_ids less than the set distance 
		--that have a minimum number of points.
		select ARRAY(select distinct (n.nearest_port_id)
			from nearest_site as n
			where n.nearest_port_dist_km < 3
			group by nearest_port_id
			having count(n.nearest_port_id) > 10) as all_sites)
		-- get the intersection of all sites and the clustering sites as common_sites,
		-- all of the sites with min points within 3km of a port as all sites
		select c.name, c.total_sites,
		cardinality(ports_3km.all_sites & c.site_ids) as common_sites,
		cardinality(ports_3km.all_sites) as all_sites
		from ports_3km, clustering_rollup as c)
	--determine the precision and recall
	select name, total_sites, common_sites, all_sites,
	common_sites::float / total_sites::float as precision,
	total_sites::float / all_sites::float as recall
	from p_r)
--update the clustering results and calcualte the precision and recall
update clustering_rollup set
common_sites = metrics.common_sites, 
all_sites = metrics.all_sites, 
precision = metrics.precision, 
recall = metrics.recall, 
f1 = (2*metrics.precision*metrics.recall) /(metrics.precision+metrics.recall)
from metrics
where clustering_rollup.name = metrics.name
;""")
conn.commit()
c.close()