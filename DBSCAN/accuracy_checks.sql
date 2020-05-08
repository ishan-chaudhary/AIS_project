select count(mmsi) from cargo_port_activity_5k

select count(*) from dbscan_results_20_4000
where clust_id >= 0

select * from (
	select mmsi, port_id,
	count(mmsi) as mmsi_count
	from cargo_port_activity_2k
	where port_id > 0
	group by mmsi, port_id
	order by mmsi_count desc) as foo
where mmsi_count > 250

	select port_id, mmsi,
	count(mmsi) as mmsi_count
	from cargo_port_activity_2k
	where port_id > 0
	group by port_id, mmsi
	order by port_id desc

select count(distinct(port_id)) as uniq_ports,
count(distinct(mmsi)) as uniq_mmsis,
count(distinct (mmsi, port_id))
from cargo_port_activity_5k

select distinct(port_id_with_most_points)
from dbscan_results_by_mmsirollup_2_500
join where distinct(port_id)
from cargo_port_activity_2k

create table round2_5_25 as
select clust_id as posit_id, geom,
ST_ClusterDBSCAN(geom, eps := .001, minpoints := 2)
over () as clust_id
from dbscan_results_by_mmsirollup_5_25
order by dbscan_results_by_mmsirollup_5_25.clust_id

select clust_id, count(*) from round2_test group by clust_id
order by count desc

select count(*) from round2_test

select id, geog,
ST_ClusterDBSCAN(geog::geometry, eps := .000005, minpoints := 2)
over () as clust_id
from ship_position_1000
order by clust_id

SELECT distinct(ST_SRID(geog)) from cargo_tanker_position

select geog::geometry from cargo_tanker_position limit 10


