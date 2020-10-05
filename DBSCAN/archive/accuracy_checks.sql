--returns all ports with the count of positions at that port within
--specified distance (currently 5km)
create table ports_5k_positions as
select 
	wpi.port_name, 
	ship_ports.nearest_site_id,
	count(ship_ports.id), 
	wpi.geom
from ship_ports as ship_ports, wpi
where ship_ports.nearest_site_id=wpi.index_no
and ship_ports.nearest_site_dist_km < 5
group by (ship_ports.nearest_site_id, wpi.port_name, wpi.geom)
order by count 




