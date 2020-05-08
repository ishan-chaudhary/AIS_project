create table cargo_tanker_mmsis as 
SELECT distinct(mmsi)
FROM ship_summary
where ship_type IN (
--cargo carriers 14,083,872 positions, 2842 mmsi
'70','71','72','73','74','75','76','77','78','79','1003','1004','1016',
--tankers 4,688,096 positions, 1199 mmsi
'80','81','82','83','84','85','86','87','88','89','1017','1024')

create table cargo_mmsis as 
SELECT distinct(mmsi)
FROM ship_summary
where ship_type IN (
--cargo carriers 14,083,872 positions, 2842 mmsi
'70','71','72','73','74','75','76','77','78','79','1003','1004','1016')



create table cargo_position as 
SELECT *
FROM ship_position
where ship_position.mmsi IN (
select * from cargo_mmsis)

select count(*) from cargo_tanker_position
select count(distinct(mmsi)) from cargo_tanker_position


SELECT sum(position_count)
FROM ship_summary
where ship_type IN (
--cargo carriers 14,083,872 positions, 2842 mmsi
'70','71','72','73','74','75','76','77','78','79','1003','1004','1016',
--tankers 4,688,096 positions, 1199 mmsi
'80','81','82','83','84','85','86','87','88','89','1017','1024')


select ship_type, (count(distinct(mmsi))) 
from ship_info
group by ship_type
order by count DESC

alter table ship_trips 
add column ship_type text

alter table ship_trips
drop column ship_type

--couldnt get this to work
insert into ship_trips (ship_type)
values (select ship_info.ship_type
from ship_info
join ship_trips
on ship_trips.mmsi = ship_info.mmsi)

--make new summary table
create table ship_summary as
select ship_info.mmsi, ship_info.ship_type,
ship_trips.position_count, ship_trips.line_length_km, ship_trips.time_diff
from ship_info, ship_trips
where ship_trips.mmsi = ship_info.mmsi

select sum(position_count), ship_type
from ship_summary
group by ship_type
order by sum DESC

select * from ship_summary 
where mmsi in (select * from cargo_tanker_mmsis)
order by position_count desc

--truncate cargo positions
create table raw_cargo_position as 
select mmsi, time, lat, lon
from cargo_position
