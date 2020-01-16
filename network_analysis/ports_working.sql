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

-- try merging back into the original table ship_position_sample
SELECT * FROM
(SELECT s.mmsi, s.time, wpi.port_name, wpi.index_no as port_id
FROM ship_position_sample AS s
JOIN wpi 
ON ST_DWithin(s.geog, wpi.geog, 2000)) as foo

SELECT pos.mmsi, pos.time, pos.geog, pos.lat, pos.lon, 
pa.port_name, port_id
FROM ship_position_sample as pos 
LEFT OUTER JOIN
port_activity_sample as pa
ON (pa.mmsi = pos.mmsi) AND
(pa.time = pos.time)





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






