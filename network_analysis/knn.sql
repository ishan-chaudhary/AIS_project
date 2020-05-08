
    -- This SQL query has two with selects and then a final select to create the new table.
    -- First create the table.  Syntax requires its creation before any with clauses.

    -- First with clause gets all positions within x meters of any port.  Note there are dupes.
    WITH port_activity as (
    		SELECT s.id, s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id,
    		(ST_Distance(s.geog, wpi.geog)) as dist_meters
    		FROM ship_position_1000 AS s
    		JOIN wpi 
    		ON ST_DWithin(s.geog, wpi.geog, 5000)
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

--this works and creates a new table
create table ship_position_ports as
with knn as (select posit.id, 
	wpi.index_no as nearest_port_id, 
	wpi.port_name as nearest_port_name, 
	(ST_Distance(wpi.geog::geometry, posit.geom)/1000) AS nearest_port_dist_km
	from ship_position_1000 as posit
	cross join lateral
	(select wpi.index_no,
	 wpi.port_name,
	 wpi.geog
	 from wpi
	 order by
	wpi.geom <-> posit.geom limit 1)
	as wpi)
select posit.id, nearest_port_id, nearest_port_name, nearest_port_dist_km
from knn
join ship_position_1000 as posit
on knn.id = posit.id
where knn.nearest_port_dist_km < 5

INSERT INTO ship_position_1000 (
	nearest_port_id,
	nearest_port_name,
	nearest_port_dist_km)
select knn.nearest_port_id, knn.nearest_port_name, knn.nearest_port_dist_km
from (select posit.id, 
	wpi.index_no as nearest_port_id, 
	wpi.port_name as nearest_port_name, 
	(ST_Distance(wpi.geog::geometry, posit.geom)/1000) AS nearest_port_dist_km
	from ship_position_1000 as posit
	cross join lateral
	(select wpi.index_no,
	 wpi.port_name,
	 wpi.geog
	 from wpi
	 order by
	wpi.geom <-> posit.geom limit 1)
	as wpi) as knn
join ship_position_1000 as posit
on knn.id = posit.id
where knn.nearest_port_dist_km < 5

select count(* from ship_position_ports
where nearest_port_dist_km < 5;
	
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