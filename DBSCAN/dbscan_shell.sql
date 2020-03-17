ALTER TABLE ship_position_1000
ADD COLUMN id SERIAL PRIMARY KEY;

CREATE TABLE dbscan_results_001_50 AS 
SELECT s.id, s.mmsi, s.lat, s.lon, port.port_name, port.port_id, 
	ST_ClusterDBSCAN(Geometry(geog), eps := .001, minpoints := 50) over () as clust_id
FROM ship_position_sample as s
JOIN port_activity_sample as port
ON s.mmsi=port.mmsi AND
s.time=port.time;


ALTER TABLE db_scan_results_001_50
ADD COLUMN geom geometry(Point, 4326);
UPDATE db_scan_results_001_50 
SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);

--need to include the port activity, mmsi, and time in this new table.
--may also recacl port distance greater than 2 km

select * from db_scan_results_001_50 limit 10;

select distinct(clust) from db_scan_results_001_50;

ALTER TABLE db_scan_results_001_50
ADD COLUMN geom geometry(Point, 4326);
UPDATE db_scan_results_001_50 
SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);