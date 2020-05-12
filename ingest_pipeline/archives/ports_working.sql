-- for testing
create table ship_test as
select * from ship_position_sample
where mmsi = '338227000';

-- Add geog column to WPI
ALTER TABLE wpi
ADD COLUMN geog geography(Point, 4326);
UPDATE wpi SET geog = Geography(ST_Transform(geom,4326));

-- Add geog column to ship_test
ALTER TABLE ship_test
ADD COLUMN geog geography(Point, 4326);
UPDATE ship_test SET geog = Geography(ST_Transform(point_geog,4326));

--add indexes
CREATE INDEX wpi_geog_idx
  ON wpi
  USING GIST (geog);
CREATE INDEX ship_test_geog_idx
  ON ship_test
  USING GIST (geog);
 CREATE INDEX ship_position_sample_geog_idx
  ON ship_position_sample
  USING GIST (geog);
 CREATE INDEX ship_position_geog_idx
  ON ship_position
  USING GIST (geog); 

--Spatial joins with geog points
CREATE TABLE port_activity_sample AS
SELECT s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id
FROM ship_position_sample AS s
JOIN wpi 
ON ST_DWithin(s.geog, wpi.geog, 2000);

-- Reduce all port activity to the first and last position at port
CREATE TABLE port_activity_sample_reduced AS
select mmsi, min(time), max(time), port_name, port_id
from port_activity_sample 
group by(mmsi,port_id, port_name) 
order by (mmsi, min(time));

select * from ship_position limit 5;

-- recraft sql query to get input for network building

CREATE TABLE port_activity AS
SELECT s.mmsi, s.time, wpi.port_name, wpi.id as port_id
FROM ship_position AS s
JOIN wpi 
ON ST_DWithin(s.geog, wpi.geog, 2000);

CREATE TABLE port_activity_reduced AS
select mmsi, min(time), max(time), port_name, port_id
from port_activity 
group by(mmsi,port_id, port_name) 
order by (mmsi, min(time));





-- This block returns the mmsi, time, port id, port name, and the distance
-- in meters to that port from the position for every position within 
-- 2000 m of a known port.
CREATE TABLE port_activity AS 
(SELECT s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id,
		(ST_Distance(s.geog, wpi.geog)) as dist_meters
			FROM ship_position_sample AS s
			JOIN wpi 
			ON ST_DWithin(s.geog, wpi.geog, 2000));

-- This block takes the port_activity table selects the min distance, and
-- joins it back to the full port_activity data to reduce duplicates when
-- many ports are within the distance
CREATE TABLE port_activity_reduced AS 
SELECT pa.mmsi, pa.time, pa.port_name, pa.port_id, t_agg.dist_meters FROM
	(SELECT mmsi, time, min(dist_meters) as dist_meters 
	FROM port_activity as pa
	GROUP BY (mmsi, time)) as t_agg, 
	port_activity as pa
	WHERE pa.mmsi = t_agg.mmsi AND
	pa.time = t_agg.time AND
	pa.dist_meters = t_agg.dist_meters;

-- This block rejoins the previously defined port activity with the
-- original position activity.  There are still duplicates though...
	CREATE TABLE port_activity_sample AS
-- We need all of the original fields from ship_position as well 
-- as the port name and port id.  
		SELECT pos.mmsi, pos.time, pa.port_name, pa.port_id
		FROM 
		ship_position_sample as pos 
		LEFT JOIN
-- this query returns all ship positions within 2000 m of a port.
-- duplicates are still possible here.
		port_activity_reduced as pa
-- we then joint the port activity (pa) with all of the ship positions
-- where the mmsi and time are equal
		ON (pa.mmsi = pos.mmsi) AND
		(pa.time = pos.time) 
		ORDER BY (pos.mmsi, pos.time);



-- This SQL query has two with selects and then a final select to create the new table.
-- First create the table.  Syntax requires its creation before any with clauses.
CREATE TABLE port_activity_sample AS
-- First with clause gets all positions within x meters of any port.  Note there are dupes.
WITH port_activity as (
		SELECT s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id,
		(ST_Distance(s.geog, wpi.geog)) as dist_meters
		FROM ship_position_sample AS s
		JOIN wpi 
		ON ST_DWithin(s.geog, wpi.geog, 2000)
-- Second with clause has a nested select that returns the closest port and groups by mmsi and time.
-- This result is then joined back to the original data.
),  port_activity_reduced as (
		SELECT pa.mmsi, pa.time, pa.port_name, pa.port_id, t_agg.dist_meters FROM
		(SELECT mmsi, time, min(dist_meters) as dist_meters 
		FROM port_activity as pa
		GROUP BY (mmsi, time)) as t_agg, 
		port_activity as pa
		WHERE pa.mmsi = t_agg.mmsi AND
		pa.time = t_agg.time AND
		pa.dist_meters = t_agg.dist_meters
)
-- We need all of the original fields from ship_position as well so this block joins the
-- results back to ALL positions, regardles if they were near a port.  
		SELECT pos.mmsi, pos.time, pa.port_name, pa.port_id
		FROM 
		ship_position_sample as pos
		LEFT JOIN
		port_activity_reduced as pa
		ON (pa.mmsi = pos.mmsi) AND
		(pa.time = pos.time) 
		ORDER BY (pos.mmsi, pos.time);



-- check queries
	
select distinct(mmsi) from (
select mmsi, time,  count (*) as total
from ship_position
group by (mmsi, time)
order by count (*) DESC) as foo
where total > 1 limit 10;

select latitude, longitude, count (*) as total
from wpi
group by (latitude, longitude)
order by count (*) DESC;
	
select distinct(mmsi) from ship_position;


			
SELECT s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id, 
wpi.geom as port_geom, s.geog::geometry as position_geom
		FROM ship_position_sample AS s
		JOIN wpi 
		ON ST_DWithin(s.geog, wpi.geog, 2000)
		where s.mmsi ='367115580' and
		s.time = '2017-01-18 23:18:51';

	(SELECT s.mmsi, s.time, s.geog as position_geog,
	 wpi.port_name, wpi.index_no as port_id, wpi.geog as port_geog
	FROM ship_position_sample AS s
	JOIN wpi 
	ON ST_DWithin(s.geog, wpi.geog, 2000))



SELECT * FROM
(SELECT  s.mmsi, s.time, wpi.port_name, 
 wpi.geom as port_geom
		FROM ship_position_sample AS s
		JOIN wpi 
		ON ST_DWithin(s.geog, wpi.geog, 2000)
		where s.mmsi ='367115580' and
		s.time = '2017-01-18 23:18:51'
		GROUP BY (s.mmsi, s.time, wpi.port_name, wpi.geom)) as foo
		ORDER BY port_geom <->  
		(SELECT s.geog::geometry from ship_position_sample AS s)
		limit 1;




CREATE TABLE port_activity_sample AS
-- We need all of the original fields from ship_position as well 
-- as the port name and port id.  
		SELECT pos.mmsi, pos.time, pa.port_name, pa.port_id, pos.geog
		FROM 
		ship_position_sample as pos 
		LEFT JOIN
-- this query returns all ship positions within 2000 m of a port
			(SELECT s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id
			FROM ship_position_sample AS s
			JOIN wpi 
			ON ST_DWithin(s.geog, wpi.geog, 2000))
		as pa
-- we then joint the port activity (pa) with all of the ship positions
-- where the mmsi and time are equal
		ON (pa.mmsi = pos.mmsi) AND
		(pa.time = pos.time) 
		ORDER BY (pos.mmsi, pos.time)




 -- This SQL query has two with selects and then a final select to create the new table.
    -- First create the table.  Syntax requires its creation before any with clauses.

    -- First with clause gets all positions within x meters of any port.  Note there are dupes.
    WITH port_activity as (
    		SELECT s.id, s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id,
    		(ST_Distance(s.geom::geography, wpi.geog)) as dist_meters
    		FROM ship_position_sample AS s, wpi as wpi
    		JOIN wpi 
    		ON ST_DWithin(s.geom, wpi.geom, {2})
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
    -- We need all of the original fields from ship_position as well so this block joins the
    -- results back to ALL positions, regardles if they were near a port.  
    		SELECT pos.id, pos.mmsi, pos.time, pos.geom::geography, pa.port_name, pa.port_id
    		FROM 
    		{0} as pos
    		LEFT JOIN
    		port_activity_reduced as pa
    		ON (pa.mmsi = pos.mmsi) AND
    		(pa.time = pos.time) 
    		ORDER BY (pos.mmsi, pos.time);""".format(source_table, destination_table, dist)
          

