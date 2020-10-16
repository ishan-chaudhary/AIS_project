select origin, (count(origin))
from cargo_edges
group by origin
order by count desc

select count(distinct(mmsi))
from cargo_edges

select count(distinct(mmsi))
from ship_ports
where nearest_site_dist_km < 5