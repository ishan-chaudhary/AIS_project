DROP TABLE IF EXISTS ship_trips;

CREATE TABLE ship_trips AS
SELECT pos.mmsi,
ST_MakeLine(pos.geom ORDER BY pos.time) AS line,
ST_Length(ST_MakeLine(pos.geom ORDER BY pos.time))/1000 as line_lengh,
COUNT(pos.geom) as count,
MIN(pos.time) as first_date,
MAX(pos.time) as last_date,
MAX(pos.time) - MIN(pos.time) as time_diff
FROM ship_position as pos
GROUP BY pos.mmsi;

select count (distinct mmsi) from ship_position;

SELECT mmsi,
line,
(ST_Length(line)/1000) as line_lengh
FROM ship_trip_sample;


from ship_position_sample as pos
group by pos.mmsi

SELECT mmsi,
line,
(ST_Length(line)/1000) as line_lengh
FROM ship_trip_sample;

