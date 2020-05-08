CREATE EXTENSION postgis;

-- Create table
CREATE TABLE IF NOT EXISTS cargo_ship_position
(
    mmsi text,
    time timestamp,
    lat numeric,
    lon numeric
)

-- add primary key
alter table cargo_ship_position 
add column id serial primary key;

-- add geom column
ALTER TABLE cargo_ship_position
add column geom geometry(Point, 4326)
-- populate with geom from lat and lon
UPDATE cargo_ship_position SET geom = ST_SetSRID(
	ST_MakePoint(lon, lat), 4326);
						  
CREATE INDEX ship_position_mmsi_idx on cargo_ship_position (mmsi);
CREATE INDEX ship_position_geom_idx ON cargo_ship_position USING GIST (geom)

--ship trips
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
 COUNT (pos.geom) as position_count,
 ST_MakeLine(pos.geom ORDER BY pos.time) AS line,
 MIN (pos.time) as first_date,
 MAX (pos.time) as last_date
 FROM cargo_ship_position as pos
 GROUP BY pos.mmsi) AS foo;