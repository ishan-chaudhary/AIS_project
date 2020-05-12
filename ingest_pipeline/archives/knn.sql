
    -- This SQL query has two with selects and then a final select to create the new table.
    -- First create the table.  Syntax requires its creation before any with clauses.

    -- First with clause gets all positions within x meters of any port.  Note there are dupes.
    WITH port_activity as (
    		SELECT s.id, s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id,
    		(ST_Distance(s.geom::geography, wpi.geog)) as dist_meters
    		FROM ship_position_1000 AS s
    		JOIN wpi
    		ON ST_DWithin(s.geom::geography, wpi.geog, 5000)
    -- Second with clause has a nested select that returns the closest port and groups by mmsi and time.
    -- This result is then joined back to the original data.
    ),  port_activity_reduced as (
    		SELECT pa.id, pa.mmsi, pa.time, pa.port_name, pa.port_id, t_agg.dist_meters FROM
    		(SELECT mmsi, time, min(dist_meters) as dist_meters
    		FROM port_activity as pa
    		GROUP BY (mmsi, time)) as t_agg,
    		port_activity as pa
    		WHERE pa.mmsi = t_agg.mmsi AND
    		pa.time = t_agg.time AND
    		pa.dist_meters = t_agg.dist_meters
    )
	SELECT * from port_activity_reduced

create table ship_position_1000 as
select * from ship_position_sample
limit 1000


select count(*) from ship_ports where mmsi = '316018616'


ALTER TABLE ship_position_1000
ADD COLUMN nearest_port_name VARCHAR,
ADD COLUMN nearest_port_id bigint,
ADD COLUMN nearest_port_dist_km float;

INSERT INTO ship_position_1000 (
	nearest_port_id,
	nearest_port_name,
	nearest_port_dist_km)
	left join
	knn where knn.id = ship_position_1000.id

--this works and creates a new table.  5/9/2019 created ship_ports
--i used geom to calc distance instead of geog.  the knn should be fine but dist
-- is wrong.
--reworked 10 may.  rerunning.
create table ship_ports as
with knn as (select posit.id, posit.mmsi, posit.time, posit.geom as ship_posit,
	wpi.index_no as nearest_port_id,
	wpi.geog as port_geog
	from cargo_ship_position as posit
	cross join lateral
	(select wpi.index_no,
	 wpi.geog
	 from wpi
	 order by
	wpi.geom <-> posit.geom limit 1)
	as wpi)
select knn.id, knn.mmsi, knn.time, knn.nearest_port_id,
(ST_Distance(knn.port_geog, knn.ship_posit::geography)/1000) AS nearest_port_dist_km
from knn
join ship_position_1000 as posit
on knn.id = posit.id

--redo of ditance cal 10 may
with full_posit as
(select posit.geom, wpi.geog, ports.id, ports.nearest_port_id
from cargo_ship_position as posit, wpi as wpi, ship_ports as ports
where posit.id=ports.id and wpi.index_no=ports.nearest_port_id
limit 10)
select (ST_Distance(full_posit.geom::geography, full_posit.geog)/1000) AS nearest_port_dist_km
from full_posit

--this took 17 hours to execute.  oof.  also, should include nearest_port_dist_km.
create temp table cargo_ship_position_temp as
select posit.id, posit.mmsi, posit.time, posit.lat, posit.lon, posit.geom,
knn.port_id_within_5k
from (select posit.id,
	wpi.index_no as port_id_within_5k,
	wpi.port_name as port_name_within,
	(ST_Distance(wpi.geog::geometry, posit.geom)/1000) AS nearest_port_dist_km
	from cargo_ship_position as posit
	cross join lateral
	(select wpi.index_no,
	 wpi.port_name,
	 wpi.geog
	 from wpi
	 order by
	wpi.geom <-> posit.geom limit 1)
	as wpi) as knn
join cargo_ship_position as posit
on knn.id = posit.id
where knn.nearest_port_dist_km < 5;

drop table cargo_ship_position;
create table cargo_ship_position as
select * from cargo_ship_position_temp;
drop table cargo_ship_position_temp;

CREATE INDEX cargo_port_mmsi_idx on ship_ports (mmsi);
CREATE INDEX cargo_port_geom_idx ON ship_ports USING GIST (geom);

select * from ship_ports
where

--analysis of new column in cargo_ship_position
with port_counts as (select nearest_port_id, count(nearest_port_id) as counts
from ship_ports
where  nearest_port_dist_km < 5
group by nearest_port_id)
select port_counts.counts, wpi.port_name
from port_counts, wpi
where port_counts.nearest_port_id=wpi.index_no
order by counts desc


select * from ship_ports
where nearest_port_dist_km < 2
limit 10


select max(nearest_port_dist_km)
from ship_ports


select nearest_port_id, avg(nearest_port_dist_km)
from ship_ports
group by nearest_port_id
order by count desc

select sum(port_id_within_5k)
from cargo_ship_position

select * from ship_position_sample where id = 678527;

CREATE INDEX ship_position_ports_dist_idx on ship_position_ports (nearest_port_dist_km)

    -- We need all of the original fields from ship_position as well so this block joins the
    -- results back to ALL positions, regardles if they were near a port.
    		SELECT pos.id, pos.mmsi, pos.time, pos.geog, pa.port_name, pa.port_id
    		FROM
    		ship_position_1000 as pos
    		LEFT JOIN
    		port_activity_reduced as pa
    		ON (pa.mmsi = pos.mmsi) AND
    		(pa.time = pos.time)
    		ORDER BY (pos.mmsi, pos.time);
