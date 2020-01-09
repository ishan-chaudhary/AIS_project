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
SELECT s.mmsi, s.time, wpi.port_name
FROM ship_position_sample AS s
JOIN wpi 
ON ST_DWithin(s.geog, wpi.geog, 2000);

-- Reduce all port activity to the first and last position at port
CREATE TABLE port_activity_reduced AS
select mmsi, min(time), max(time), port_name 
from port_activity 
group by(mmsi,port_name) 
order by (mmsi, min(time));

select * from ship_position limit 5;

-- recraft sql query to get input for network building

CREATE TABLE port_activity_test AS
SELECT s.mmsi, s.time, wpi.port_name
FROM ship_test AS s
JOIN wpi 
ON ST_DWithin(s.geog, wpi.geog, 2000);

CREATE TABLE port_activity_test_reduced AS
select min(time) as arrive, max(time) as depart, max(time) - min(time) as time_diff, 
count(mmsi), port_name 
from port_activity_test 
group by port_name
order by min(time);



