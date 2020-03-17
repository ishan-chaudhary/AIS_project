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