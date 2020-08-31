--return clust_id, total points, avg_geom, nearest_port_id, site name and site geom
with final as(
	-- creates summary for the clustering results joined with original poistion info
	with summary as (
		select c.dbscan_0_25_25 as clust_result, pos.uid as uid, count(pos.id) as total_points, 
		ST_Centroid(ST_union(pos.geom)) as avg_geom
		from uid_positions as pos, clustering_results as c
		where c.id = pos.id
		and c.dbscan_0_25_25 is not null
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
insert into clustering_rollup (total_clusters, avg_points, average_dist_nearest_port,
							  site_names, site_ids)
select count(final.clust_id), 
avg(final.total_points), 
avg(final.nearest_port_dist_km),
array_agg(distinct(final.site_name)),
array_agg(distinct(final.nearest_port_id))
from final

SELECT *
  FROM information_schema.columns
 WHERE table_schema = 'public'
   AND table_name   = 'clustering_results'
   AND column_name != 'id'


--find all ports within 5km of each point
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


select name, total_clusters, avg_points, average_dist_nearest_port,
total_sites, precision, recall
from clustering_rollup order by f1


cardinality(ports_3km.all_ports & c.site_ids)::float  / c.total_sites::float as precision,
c.total_sites::float / cardinality(ports_3km.all_ports)::float as recall


create extension intarray

select column1, column1 & ARRAY[3,4,8] as elements
from table1





--find all ports within 5km of each point
select n.nearest_port_id, sites.port_name, count(n.nearest_port_id) as pos_count
from nearest_site as n, sites
where nearest_port_dist_km < 3
and sites.site_id = n.nearest_port_id
group by nearest_port_id, sites.port_name
having count(n.nearest_port_id) > 10
order by pos_count



--get clust_id, uid, id, lat, lon from clustering results
select c.dbscan_0_25_25 as clust_id, pos.uid, pos.id, pos.lat, pos.lon
from uid_positions as pos, clustering_results as c
where c.id = pos.id
and c.dbscan_0_25_25 is not null
limit 100