-- add postgis extension
CREATE EXTENSION postgis;

-- make point_geog column
ALTER TABLE ship_position 
ADD COLUMN geog geography (Point, 4326);

-- populate geog column
UPDATE ship_position SET geog = ST_SetSRID(
	ST_MakePoint(lon, lat), 4326);
	
-- create indices on ship_positions
CREATE INDEX ship_position_mmsi_idx on ship_position (mmsi);
CREATE INDEX ship_position_geog_idx
  ON ship_position
  USING GIST (geog);
  

-- clear table
DROP TABLE IF EXISTS ship_trips;

-- generate new ship trips table
CREATE TABLE ship_trips AS
SELECT mmsi,
		position_count,
		line,
		ST_Length(geography(line))/1000 AS line_length_km,
		first_date,
		last_date,
		last_date - first_date as time_diff
FROM (
 SELECT pos.mmsi,
 COUNT (pos.geog) as position_count,
 ST_MakeLine(pos.geog ORDER BY pos.time) AS line,
 MIN (pos.time) as first_date,
 MAX (pos.time) as last_date
 FROM ship_position as pos
 GROUP BY pos.mmsi) AS foo;
 
-- create mmsi index on ship_trips
CREATE INDEX ship_trips_mmsi_idx on ship_trips (mmsi);




-- SAMPLING PRACTICE
-- clear table
DROP TABLE IF EXISTS ship_position_sample;
DROP TABLE IF EXISTS ship_trips_sample;

-- make a sample ship_position table
CREATE TABLE ship_position_sample AS
SELECT * FROM 
(SELECT mmsi, time, lat, lon FROM ship_position limit 500000) as Foo;

-- create an index on mmsi to see if its faster than the ~30 mins without
CREATE INDEX mmsi_index_ship_position on ship_position (mmsi);


-- make point_geog column
ALTER TABLE ship_position_sample 
ADD COLUMN point_geog geometry (Point, 4326);

UPDATE ship_position_sample SET point_geog = ST_SetSRID(
	ST_MakePoint(lon, lat), 4326);
	
SELECT geography(point_geog) from ship_position_sample limit 10;

-- create an index on mmsi to see if its faster than the ~30 mins without
CREATE INDEX mmsi_index_ship_position_sample on ship_position_sample (mmsi);

-- generate new ship trips table
CREATE TABLE ship_trips_sample AS
SELECT mmsi,
		position_count,
		line,
		ST_Length(geography(line))/1000 AS line_length_km,
		first_date,
		last_date,
		last_date - first_date as time_diff
FROM (
 SELECT pos.mmsi,
 COUNT (pos.point_geog) as position_count,
 ST_MakeLine(pos.point_geog ORDER BY pos.time) AS line,
 MIN (pos.time) as first_date,
 MAX (pos.time) as last_date
 FROM ship_position_sample as pos
 GROUP BY pos.mmsi) AS foo;
 
CREATE TABLE ship_position_sample AS
SELECT * FROM ship_position
WHERE mmsi = '316024713';
 
