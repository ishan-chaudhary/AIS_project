    -- speed thest for 10 km and 1000 sample
	SELECT id, lat, lon, 
    ST_ClusterDBSCAN(Geometry(geog), eps := 0.00157, 
    minpoints := 1000) over () as clust_id
    FROM ship_position_sample;
	
	--local
	--4000mb shared buffers, 2000mb work mem: 23mins 29secs
	--8000mb shared buffers, 2000mb work mem: 22mins 28secs
	
	show work_mem;
	-- aws
	--160mb shared buffers, work mem 400mb