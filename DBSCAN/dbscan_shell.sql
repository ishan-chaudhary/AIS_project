CREATE TABLE db_scan_results_001_50 AS  
	SELECT id lat, lon, 
			ST_ClusterDBSCAN(Geometry(geog), eps := .001, minpoints := 50) over () as clust
FROM ship_position_sample;


select * from db_scan_results_001_50 limit 10;

select distinct(clust) from db_scan_results_001_50;

ALTER TABLE db_scan_results_001_50
ADD COLUMN geom geometry(Point, 4326);
UPDATE db_scan_results_001_50 
SET geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326);