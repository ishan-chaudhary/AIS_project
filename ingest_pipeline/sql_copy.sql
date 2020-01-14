--try importing csvs directly into SQL
DROP TABLE IF EXISTS imported_ais;
CREATE TABLE imported_ais (
  	mmsi 			text,
	time			timestamp,
	lat				numeric,
	lon				numeric,
	sog				varchar,
	cog				varchar,
	heading			varchar,
	ship_name		text,
	imo				varchar,
	callsign		varchar,
	ship_type		text,
	status			varchar,
	len				varchar,
	width			varchar,
	draft			varchar,
	cargo			varchar);
	
COPY imported_ais 
FROM '/Users/patrickmaus/Documents/projects/AIS_data/2017/AIS_2017_01_Zone09.csv'
WITH (format csv, header);

INSERT INTO ship_position (mmsi, time, lat, lon)
SELECT mmsi, time, lat, lon FROM imported_ais;
	
INSERT INTO ship_info (mmsi, ship_name, ship_type)
SELECT DISTINCT mmsi, ship_name, ship_type from imported_ais;

DELETE FROM imported_ais;
  