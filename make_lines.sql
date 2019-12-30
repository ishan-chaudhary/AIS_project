-- add postgis extension
CREATE EXTENSION postgis;

-- make point_geog column
ALTER TABLE ship_position 
ADD COLUMN point_geog geometry (Point, 4326);

-- populate point_geog column
UPDATE ship_position SET point_geog = ST_SetSRID(
	ST_MakePoint(lon, lat), 4326);

-- clear table
DROP TABLE IF EXISTS ship_trips;

-- generate new ship trips table
CREATE TABLE ship_trips AS
SELECT mmsi,
		position_count,
		line,
		ST_Length(line)/1000 AS line_length,
		first_date,
		last_date,
		last_date - first_date as time_diff
FROM (
 SELECT pos.mmsi,
 COUNT(pos.point_geog) as position_count,
 ST_MakeLine(pos.point_geog ORDER BY pos.time) AS line,
 (select MIN(pos.time) from ship_position as pos) as first_date,
 (select MAX(pos.time) from ship_position as pos)  as last_date
 FROM ship_position as pos
 GROUP BY pos.mmsi) AS foo;
 
 -- original query
SELECT pos.mmsi,
ST_MakeLine(pos.geom ORDER BY pos.time) AS line,
ST_Length(ST_MakeLine(pos.geom ORDER BY pos.time))/1000 as line_lengh,
COUNT(pos.geom) as count,
MIN(pos.time) as first_date,
MAX(pos.time) as last_date,
MAX(pos.time) - MIN(pos.time) as time_diff
FROM ship_position as pos
GROUP BY pos.mmsi;
 
