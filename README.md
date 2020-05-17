# AIS_project Overview

This project's goals are to:
- Gain experience building out data pipelines for large, geospatial datasets myself
- Learn more about PostGIS and QGIS through practical application
- Practice good software engineering practices in my data science projects
- Experiment with different ways of identifying and evaluating clusters in space and time
- Developing optimal implementations of oft-used functions, such as finding distances between two lists
- Translating dense spatial-temporal data into networks and analyzing them
- Analyze network analysis, including machine learning for prediction
- Conduct effective timeseries analysis and build appropriate forecasting models

The project has four major phases.
  1. Data ingest, cleaning, and analysis.
  2. Cluster creation and evaluation.
  3. Network analysis and prediction.
  4. Time series analysis and prediction.

  Each phase has multiple individual projects.  This readme will try to document the project and track development.

## Data Sources
  Easy to scrape website:
  https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2017/index.html
  Pretty website:
  https://marinecadastre.gov/ais/
  World Port Index (loaded using QGIS into Post GIS)
  https://msi.nga.mil/Publications/WPI


## Required Software

  - PostGreSQL with post GIS
  - PG Admin 4
  - QGIS
  - Spyder

  Python Packages Used (Current 24 April 2020)

  Database management
  - psycopg2
  - sqlalchemy

  Data cleaning and Analysis
  - Pandas
  - numpy

  Visualization and Plotting
  - matplotlib

  File management
  - glob
  - os

  Time tracking
  - datetime

  Model Building and calculations
  - from sklearn.neighbors import BallTree
  - from sklearn.metrics.pairwise import haversine_distances
  - from sklearn.cluster import DBSCAN

  Customized Modules
  The analysis in this project uses functions from the gsta module, which stands for GeoSpatial Temporal Analysis module.  The gsta_config file is not share on the repo and stores server information including passwords and credentials.  This replaces the aws_credentials module.

# Data Ingest, Cleaning, and Analysis

  The AIS data is large.  January 2017 data fro the US is 25 gigabytes.  The entire year could be about 300 gigabytes.  There are two options here.  The first is to put it all in the cloud with all the raw data and cleaned tables.  The second is to  store all csvs as zipped files, process and clean he data in chunks to a database, and create a summary for every vessel in the data.  Then, we can sample the raw positions and conduct cluster analysis on those samples.  Armed with our summary analysis of the entire data, we can then ensure that our samples are representative of different patterns.

  We will first pursue the latter option, but may eventually use AWS to spin-up a PostGres instance for the entire dataset.

  Current implementation (16 January 2020) uses a source directory as an input and iterates through all the csv files in that directory.  Future improvements can allow filtering based on month and zone.  Additionally, can eventually add a webscraper component.


## Ingest Pipeline Summary
  There are two options for ingesting data and building out our database tables.  The first is to use the ingest_SQL_script to manually create the tables, build indices, populate sample tables and build out the summary tables.  This works best for ingesting one csv, such as the sample of Cargo vessels used for much of this analysis.  For loading in multiple csvs, it will be more efficient to use the Python scrip ingest_script_prod.py.

  In both versions, the sequencing is very similar.  

  - Create a new database and then build a Post GIS extension.  
  - Create a connection to the db
  - Drop any existing tables using a custom 'drop_table' function.  (Note, in prod the actual calls are commented out for safety.)
  - Create a "dedupe_table" function for the ship_info table.
  - Create the tables 'imported_ais' to hold the original copied data, the ship_info table that includes mmsi, ship name and ship type, the ship_position table with location data for each position, and the wpi table that has info from the world port index for each major port.
  - The "parse_ais_SQL" function ingests the raw csv file to imported_ais and then selects the relevant data for the ship_info and ship_position table.
  - The "make_ship_trips" function creates a ship_trips table that reduces each ship's activity in the ship_position table to summary stats, including a line of the ship's voyage.
  - All activities are recorded to a processing log ("proc_log") with the date appended.  "function_tracker" function executes key functions, logs times, and prints updates to the console.

## Reducing Raw position data to lines
  Unfortunately analysis against the entire corpus is difficult with limited compute.   However, we can reduce the raw positions of each ship into a line segment that connects each point together in the order of time.  Because there are sometimes gaps in coverage, a line will not follow the actual ships path and "jump".  The "make_ship_trips" function in the ingest pipeline uses geometry in PostGIS to connect all positions per ship, but this is not accurate as the coverage is not consistent.

  We also will want to add ship_type into this table to assist in clustering later down the road.

## Analyze Summarized Ship trips
  Since we don't want to test all of our algorithm development against the entirety of the data, we need to select a sample of MMSI from the entire population to examine further.  The Jupyter Notebook "ships_trips_analysis" parses the ships_trips table and analyzes the fields.  We can use it to select a sample of MMSIs for further analysis.  The notebook builds several graphs and examines the summary of each MMSI's activity.  After filtering out MMSIs that don't travel far, have very few events, or travel around the globe in less than a month and are possible errors, the notebook exports a sample (250 now) to a csv file.  The notebook also writes the ship_trips sample to a new table, "ship_trips_sample" directly from the notebook using SQL Alchemy.

## Create Samples for Further analysis
  Python script "populate_sample" takes the csv output of the sample MMSIs from the Jupyter Notebook "ships_trips_analysis" and adds all positions from the "ship_position" table to a new "ship_position_sample" table.  It also makes a "ship_trips_sample" table from the full "ship_trips" table.

## Port Activity tables creation
  Creates a table that includes all of a ship's position when the ship's positions are within X meters of a known port.  

  The WPI dataset has  duplicate port locations.  Specifically, the exact same geos are used twice by two different named and indexed ports 13 times.  I naively dropped duplicates on the lat and lon columns to resolve.  No order is guaranteed, so this is a risk for reproducibility.  TODO: EIther order the drops in some way or record the 13 ports removed for documentation.


## Table Summary for ingest pipeline
  imported_ais --> ship_position --> ship_trips --> port_activity

## Lessons Learned
### Using PostGreSQL COPY
  This is so much faster than iterating through chunks using pandas
  Using Pandas and iterating through 100000 rows at a time on a sample csv of 150 mb took ~2 mins.  By using copy to create a temp table and then selecting the relevant info to populate the ship_info and ship_position table, the total time was reduced to 25 seconds.

### A note on Spatial Indices
  I first failed to create spatial indices at all.  This led to a 2 minute 45 second spatial join between a table with 9265 positions and the WPI table with 3630 ports.  By adding a spatial index with the below syntax to both tables, the query then took only 124 msec.  Running the same query against over 2 million rows in the sample data took 2.6 seconds.

  CREATE INDEX wpi_geog_idx
  ON wpi
  USING GIST (geog);
  CREATE INDEX ship_test_geog_idx
  ON ship_test
  USING GIST (geog);

#### Notes on Visualizing in QGIS

  Large numbers of points within a vector layer can severely impact rendering performance within QGIS.  Recommend using the "Set Layer Scale Visibility" to only show the points at a certain scale.  Currently I found a minimum of 1:100000 appropriate with no maximum.  Ship trips can have no scale visibility because they render much faster.


# Clustering
## DBSCAN Background and Implementation
  To actually cluster the positions in an attempt to find ports, I used Density-based spatial clustering of applications with noise, or DBSCAN.  Originally I hoped to use Scikit-Learn’s implementation.  But unfortunately, because of the size of the data, DBSCAN clustering of the 2 million positions consistently crashes my Python kernel in Jupyter Notebook and Spyder.  Instead, I used PostGres’ native DBSCAN algorithm which does not need to hold all of the data in memory.  I used both my own local PostGres instance for validation and a PostGres cluster stood up in Amazon Web Services (AWS).

### Clustering Entire Dataset or By Unique ID
  Originally I intended to cluster the entire dataset so the unique ID, or MMSI, for each vessel was disregarded.  If five ships had 10 position reports each, the DBSCAN algorithm would see 50 points in a given space.  My goal was to treat all ship position reports equally but I found several problems with this approach.  

  The first issue was actually processing that much data and time complexity.  Requiring to conduct DBSCAN against the entire dataset requires either smartly iterating over the dataset or holding it all in RAM.  I was unable to execute Sciki-Learn’s implementation because it could not process all original 2 million sample points in memory.  PostGres can move over each position in the table since it holds all the data within the database.  Relatedly, time complexity becomes a significant problem for large N.  In a worst-case scenario with a poorly chosen epsilon value, DBSCAN run-time complexity is O(N^2).  During our search for the most effective hyperparameters, we will almost certainly choose bad epsilon values, causing us to pay a significant penalty in processing time.

  The other issue with using the entire dataset instead of a by unique ID approach is the likelihood of future activity affecting our chosen hyperparameters.  With the entire dataset, our approach with DBSCAN will find effective hyperparameters for that available dataset.  For our case, this is just one month, January 2017.  If we added another month of data and similar patterns occurred, the density would double in certain areas, causing our well-tuned hyperparameters to reduce in effectiveness.

  The goal of this project is to identify ports where cargo ships stop, resulting in a denser number of position reports.  If instead of processing the entire dataset and clustered each individual ship’s positions, we would hope to find numerous clusters from different ships at or near ports.  Therefore I altered my approach to implement a double-clustering which takes the output of the by ship clustering and clusters again to find clusters of clusters.  Ideally, these clusters of clusters will represent the ports we are trying to find.

  There are several advantages to this approach.  First, it is less sensitive to additional data aggregating over time.  Since we are not concerned if a port has two clusters or 30 clusters, we can set the second cluster minimum samples’ hyperparameter low, which will allow a supercluster to form with just a small number of ship clusters present.  As more data is added over time, new clusters from ships active at the port will be created and included in the supercluster.  As long as the first round of clustering has well-tuned hyperparameters that prevent new by ship clusters from forming close to the port (but not in the port), there is relatively little risk of a “walking” cluster.

  Another advantage is that it can screen out false positives when a ship loiters outside a port.  Frequently, our approach clusters ships outside a port as they await clearance to enter and dock.  Additionally, a ship anchored out of a port can cause a false positive.  If we use a minimum sample size large enough, we can avoid these false positives because their density for any given time sample should be lower than a port’s density.

## PostGres Implementation
  Initial results of the PostGres DBSCAN implementation were promising.  The ST_ClusterDBSCAN function takes a geometry column, the epsilon value, the minpoints, and uses a window function to move over the entire table.  Individual points are assigned a cluster id (‘clust_id’ feature in my implementation) if they belong to a cluster and NULL if not.  To save space and decrease write times, I only wrote points associated to a cluster back to the dataframe.  

  A major advantage of PostGres was the use of server resources to move across an entire table and not have to hold it in RAM.  This allowed the PostGres implementation to execute DBSCAN across the sample data without crashing.  However, it is recommended to tune the “work_mem” and the “shared_buffers” setting on your PostGres server.  These values are normally set to a low default and increasing them based on available memory on your server or computer can dramatically improve performance.  

  >  work_mem (integer)
    Specifies the amount of memory to be used by internal sort operations and hash tables before writing to temporary disk files. The value defaults to four megabytes (4MB).
    shared_buffers (integer)
    Sets the amount of memory the database server uses for shared memory buffers. The default is typically 128 megabytes (128MB),
    -PostGres Documentation

  The PostGres implementation had a major disadvantage in its function.  ST_ClusterDBSCAN only accepts a geometry position, not a geography position.  Therefore, the unit of distance calculation is degrees of latitude and longitude rather than a fixed geodesic distance.  Although this has little implications for clustering in a small area, when the function is applied to data spread across the coastal US there can be significant differences.  

  Degrees of latitude are about 69 miles (111 km), with little variability across the earth.  Degrees of longitude vary significantly.  At the equator, a degree of longitude is also about 69 miles (111 km), but the distance reduce to zero as the converge to the North and South Poles.  For examples, at 40 degrees N (the latitude of New York City) a degree of longitude is 53 miles (85 kilometers).  However at 25 degrees north (the latitude of Miami) a degree of longitude is 63 miles (101 km).  Therefore, positions at higher latitudes will cluster at lower real-world distance given the same epsilon value in degrees.  Because of the serious risk of bias in this approach, PostGres DBSCAN should not be used when comparing clusters across significant latitudes.

## Scikit-Learn Implementation
  Using Scikit-Learn has several advantages and disadvantages.  The major disadvantage is that the data to be clustered must be extracted from the database and the results written back to the database.  Originally, this caused problems with too large of samples being computed in memory and excessive delays in writing the results back to the database.  We have already discussed the solution to holding too much memory in data, and now will discuss several steps in speeding up the write process.

  The first improvement increased speed and saved memory space.  Instead of writing back all the points used in clustering, only points assigned to a cluster were written back to the database.  This creates significant improvements in speed.  The second improvement was using SQLAlchemy’s “multi” method to write the final pandas dataframe back to the databse.  The default for SQLALchemy is to write one row at a time and the “multi” method provides a significant speed-up. There are additional ways to further improve the speed, such as moving away from pandas dataframes and using tuples or dictionaries to store the data before being written to the database.

## Accuracy
  To help identify activity near known ports, I downloaded and ingested the World Port Index’s list of ports worldwide.  Once this data was in the PostGres database, I used PostGIS’s KNN function to find the nearest port to each of the ship positions, and labeled each point with that port’s name.  I then used ST_Distance with the distance type as geography to calculate the distance to each of these ports.  This query is found in the ingest_script.sql and takes about 17 hours to perform against the 14 million cargo ship position reports.  

  Once we have this data, stored in our database as “ship_ports”, we can create a new table which lists each port and the count of position reports within any given distance.  I created “ports_5k_positions” and “ports_5k_sample_postions” using this approach with a distance of 5 kilometers, which gives us an approximate set of “correct” ports that should be found using any clustering method.

  There is one subjective factor to consider here.  How many position reports should be within the given distance to make a port a “correct” target?  It is feasible that a ship only transited nearby a port and did not stop, leading to the port being identified as a correct target.  However, clustering that low level of activity is not actually the goal of this approach and is not likely to be successfully identified.  In the interest of assigning an acceptable cutoff, I created an argument called “noise_filter” when calculating the statistics, which establishes a minimum number of position reports at a port for it to be determined a valid target.  This is a purposefully conservative filter and likely creates a lower recall percentage.

  After much consideration and testing, I decided the most effective way to measure the correctness of each approach and set of hyperparameters was to use precision and recall.  Generally, precision is the proportion of selected items are relevant while recall is the proportion of relevant items selected. Additionally, we can use the harmonic mean of the precision and recall to yield the F-measure, a single metric we can use to evaluate how each model performed.  

  Since we are not concerned with correctly labeling every single position to the “correct” port and only concerned with developing a model that accurately clusters positions at ports, the best approach is to look at the set of ports found by this clustering approach and compare it to the set of ports that have ship activity nearby that we have already labeled as “correct”.  A strength of this approach is that it limits the penalty on false positives that may be ports but are not in our gold-standard, labeled “correct” dataset.  In fact, analysis of the data identifies multiple cases where clusters are identified at established ports that are not listed in the WPI.  

  Therefore In our use case, we will define precision as the proportion of ports identified by our clustering approach
  1)	Whose average cluster position is within 5 km of a known port and
  2)	Whose nearest port has position reports within 5 km and
  3)	the count of those positions is above the noise filter
  divided by the total number of ports identified by the clustering approach.  Recall will be the proportion of ports with the same characteristics as above divided by the count of all ports labeled as “correct”.

  params	eps_km	min_samples	numb_clusters	Average cluster count	Average nearest port from center	Average dist from center	Average max dist from center	f_measure	precision	recall
  0.25_5	0.25	5	422	200.953	41.613	1.456	3.322	0.537	0.468	0.63
  0.5_10	0.5	10	192	442.156	33.266	2.62	6.688	0.535	0.491	0.587
  0.25_10	0.25	10	188	402.09	41.328	0.78	2.342	0.522	0.522	0.522
  0.25_100	0.25	100	49	1429.122	9.643	0.09	1.114	0.519	0.6	0.457
  0.5_100	0.5	100	48	1473.667	9.818	0.173	2.165	0.519	0.6	0.457
  2.0_25	2.0	25	117	799.231	33.002	7.219	19.252	0.514	0.444	0.609
  0.25_25	0.25	25	70	1024.371	15.23	0.466	1.875	0.512	0.55	0.478
  0.25_50	0.25	50	52	1351.404	11.448	0.208	1.363	0.512	0.583	0.457
  1.0_100	1.0	100	51	1432.392	12.755	0.88	6.281	0.512	0.583	0.457

  The table above shows the top ten results using our sample of 28 different cargo ships.  Notably, the best performing hyperparameters were .25 km for epsilon with a min_sample of 5.  This is close to the low end of our sample space where we conducted our grid search and would suggest we expand our space for grid search.  However, closer analysis says that lower hyperparameters would fail to cluster positions accurately.  First notice the high number of clusters, as well as the relatively large average distance from the center point of the cluster.  This suggests that the cluster is “walking”, which can be confirmed by plotting the data.  



  Sample Walkthrough
  The process is as follows:

  1.	Run the “execute_dbscan” function against the sample data (“ship_positions_sample”) with a wide-range of hyperparameters.  
  2.	Run the “analyze_dbscan” function against the resulting output, which produces summaries, precision, recall, and the F-measure.  
  3.	Investigate the results and plot the data visually.  Watch for walking clusters indicating a too-low threshold.  This approach looks for the smallest set of hyperparamters that produces clusters at or near ports that do not walk.  Multiple clusters at a single port are fine because we will cluster them together in the next phase.
  4.	Identify the 2-5 best performing sets of hyperparameters using the f-measure score.
  5.	Run the “execute_dbscan” function with the 2-5 hyperparameters against the full data set.  This can take hours/days.
  6.	Run the “analyze_dbscan” against the resulting output, which produces the average latitude and longitude for each cluster.
  7.	Run the “execute_dbscan” function with a wide-range of hyperparameters to cluster the clusters from the last round to super clusters.
  8.	Run the “analyze_dbscan” function against the results from the second round of clustering.



  Run the “execute_dbscan” function against the sample data (“ship_positions_sample”) with a wide-range of hyperparameters.  
  To evaluate how well the data clusters, we will use a range of epsilon distances and minimum samples.  A reasonable maximum epsilon is only a few kilometers, since ships frequently transmit their position and move relatively slowly.  We will search up to 7km, a very large distance that would be covered by a ship at cruising speed in about 15-20 minutes.  For the minimum distance, we will use a relatively small distance of .25km.  Because the shape of a DBSCAN cluster is not set and can be anything, a low threshold has a risk of “Walking”, where the cluster continues to grow with each successive position report of a ship and never ends.  We will examine such an example further along in this walkthrough.


  Run the “analyze_dbscan” function against the resulting output, which produces summaries, precision, recall, and the F-measure.  
  The output of this function is below.  It lists the total number of clusters, the average number of positions per cluster, the average distance to the nearest port for each cluster, the F-measure, precision and recall.  This table can be an effective tool for identifying hyperparameters that perform well against the data.  For example, parameters that have a very high average nearest port are likely to have numerous clusters far from known points, suggesting those hyperparameters lead to numerous false positives.

  Also, we will want to take a more comprehensive view than “maximize F-measure.”  For example, 10 different iterations have identical F-measure, precision, and recall but vary in number of clusters from 407 to 61.  We are interested in a set of hyperparameters that best identify “port-like” activity, so a set of hyperparameters that produce fewer clusters with similar accuracy metrics will be preferred over a set that produces more clusters.  Finally, the average nearest port metrics is very helpful in finding a set of hyperparameters that consistently finds clusters near known ports.

  params	eps_km	Min samples	Numb clusters	Average cluster count	Average nearest port from center	f_measure	precision	recall
  0.25_2	0.25	2	1447	45.424	12.163	0.644	0.683	0.609
  0.25_5	0.25	5	407	152.373	12.651	0.643	0.711	0.587
  0.25_7	0.25	7	281	213.537	13.219	0.643	0.711	0.587
  0.25_10	0.25	10	166	347.765	14.133	0.593	0.686	0.522
  0.25_25	0.25	25	60	896.733	7.556	0.587	0.759	0.478
  0.25_50	0.25	50	44	1189.295	3.9	0.629	0.917	0.478
  0.25_100	0.25	100	40	1282.625	4.118	0.588	0.909	0.435
  0.25_250	0.25	250	24	1930.125	3.805	0.484	0.938	0.326
  0.25_500	0.25	500	10	3916.9	5.053	0.321	0.9	0.196
  0.5_2	0.5	2	655	103.992	13.357	0.659	0.69	0.63
  0.5_5	0.5	5	223	298.574	14.15	0.643	0.711	0.587
  0.5_7	0.5	7	180	365.189	11.697	0.643	0.711	0.587
  0.5_10	0.5	10	159	406.094	12.36	0.643	0.711	0.587
  0.5_25	0.5	25	92	649.674	11.191	0.575	0.676	0.5
  0.5_50	0.5	50	54	1030.407	5.745	0.622	0.821	0.5
  0.5_100	0.5	100	44	1221.068	4.007	0.629	0.917	0.478
  0.5_250	0.5	250	29	1741.069	3.67	0.531	0.944	0.37
  0.5_500	0.5	500	17	2695.235	4.183	0.407	0.923	0.261
  1.0_2	1.0	2	418	166.074	16.646	0.667	0.64	0.696
  1.0_5	1.0	5	125	547.424	13.385	0.635	0.692	0.587
  1.0_7	1.0	7	110	619.018	13.04	0.635	0.692	0.587
  1.0_10	1.0	10	103	655.932	11.566	0.643	0.711	0.587
  1.0_25	1.0	25	83	789.578	10.474	0.627	0.703	0.565
  1.0_50	1.0	50	71	881.225	9.402	0.59	0.719	0.5
  1.0_100	1.0	100	55	1056.873	7.023	0.613	0.793	0.5
  1.0_250	1.0	250	31	1688.258	3.629	0.554	0.947	0.391
  1.0_500	1.0	500	17	2756.235	4.183	0.407	0.923	0.261
  2.0_2	2.0	2	649	109.206	22.256	0.667	0.623	0.717
  2.0_5	2.0	5	94	737.606	15.162	0.628	0.675	0.587
  2.0_7	2.0	7	87	793.966	14.672	0.635	0.692	0.587
  2.0_10	2.0	10	81	849.815	14.099	0.635	0.692	0.587
  2.0_25	2.0	25	72	945.569	10.036	0.643	0.711	0.587
  2.0_50	2.0	50	65	1023.169	9.143	0.6	0.706	0.522
  2.0_100	2.0	100	57	1131.509	8.506	0.59	0.719	0.5
  2.0_250	2.0	250	36	1594.806	4.015	0.559	0.864	0.413
  2.0_500	2.0	500	18	2679.056	4.065	0.433	0.929	0.283
  3.0_2	3.0	2	1045	69.178	28.138	0.654	0.586	0.739
  3.0_5	3.0	5	89	783.292	13.176	0.628	0.675	0.587
  3.0_7	3.0	7	83	837.313	14.413	0.628	0.675	0.587
  3.0_10	3.0	10	74	935.743	12.953	0.635	0.692	0.587
  3.0_25	3.0	25	66	1041.409	10.681	0.643	0.711	0.587
  3.0_50	3.0	50	63	1079.381	9.8	0.634	0.722	0.565
  3.0_100	3.0	100	55	1211.109	8.539	0.59	0.719	0.5
  3.0_250	3.0	250	41	1492.22	6.716	0.543	0.792	0.413
  3.0_500	3.0	500	23	2281.087	4.593	0.444	0.824	0.304
  5.0_2	5.0	2	1881	39.865	33.433	0.624	0.54	0.739
  5.0_5	5.0	5	111	634.658	18.162	0.614	0.643	0.587
  5.0_7	5.0	7	79	885.57	12.721	0.628	0.675	0.587
  5.0_10	5.0	10	70	995.914	12.307	0.628	0.675	0.587
  5.0_25	5.0	25	63	1099.254	11.406	0.643	0.711	0.587
  5.0_50	5.0	50	58	1187.517	9.949	0.634	0.722	0.565
  5.0_100	5.0	100	53	1282.642	8.932	0.608	0.727	0.522
  5.0_250	5.0	250	39	1647.41	7.542	0.535	0.76	0.413
  5.0_500	5.0	500	25	2266.56	4.54	0.444	0.824	0.304
  7.0_2	7.0	2	2688	29.01	36.692	0.625	0.53	0.761
  7.0_5	7.0	5	142	500.789	21.318	0.629	0.651	0.609
  7.0_7	7.0	7	92	765.902	15.455	0.614	0.643	0.587
  7.0_10	7.0	10	70	1000.071	12.767	0.628	0.675	0.587
  7.0_25	7.0	25	61	1140.672	9.71	0.643	0.711	0.587
  7.0_50	7.0	50	59	1174.729	10.799	0.634	0.722	0.565
  7.0_100	7.0	100	52	1319.462	8.6	0.608	0.727	0.522
  7.0_250	7.0	250	40	1643.225	7.582	0.535	0.76	0.413
  7.0_500	7.0	500	27	2220.333	6.917	0.462	0.789	0.326


  Investigate the results and plot the data visually.
   Watch for walking clusters indicating a too-low threshold.  This approach looks for the smallest set of hyperparamters that produces clusters at or near ports that do not walk.  Multiple clusters at a single port are fine because we will cluster them together in the next phase.



  The long streak of position reports are all components of a small number of clusters when running DBSCAN with an epsilon distance of 0.25km and a minimum sample size of 25.  Each position report is close enough to others in great enough size to exceed the minimum sample size of 5, which allows the cluster to continue “walking” with the ship’s movement.  This cluster’s hyperparameters are far too low and anything lower than these values will lead to even more extreme “Walking”.

  On the other hand, the graphic below shows the same area with the same value for epsilon of .25 km but with a higher minimum sample hyperparameter of 100.  The walking clusters are gone, but clusters at and near the major ports of Miami and Port Everglades remain.  This underscores the importance of always plotting our results to see what the data is doing.  It also suggests that in addition to the F-measure, we should compare the ratio between the average distance and maximum distance from the center of the cluster with the epsilon distance.  If either measurement are multiple factors of the epsilon distance, it is likely that we are dealing with a walking cluster.




  Identify the 2-5 best performing sets of hyperparameters using the f-measure score.
  For this walkthrough, we choose to use two sets of hyperparameters that returned low average distances to known ports, 0.5km for epsilon for both, and 50 and 100 minnimum samples.  These are not the best hyperparameters, but will be good examples for the rest of this walkthrough.

  Run the “execute_dbscan” function with the 2-5 hyperparameters against the full data set.  This can take hours/days.
  The output of this process is a table of 7.4 million rows for [.5, 50] and 7.2 million rows for [.5,100].  Each table only contains the positions identified as part of a cluster and includes the unique ID for each position, the latitude and longitude, the cluster ID assigned by DBSCAN, the geometry which can be plotted in QGIS, and the MMSI.  The size of this data makes it difficult to actually plot the data in QGIS though.

  Run the “analyze_dbscan” against the resulting output, which produces the average latitude and longitude for each cluster.

  A sample from the first five rows of the second clustering of [.5, 50] is below.  The first row is the first cluster found (cluster 0) for MMSI 123456789, as shown by the clust_id which is the concatenation of the two different values.  We can see in this data the key metrics that will be rolled-up in the summary.

  	Clust id	Average lat	Average lon	Nearest portid	Nearest port dist	Min dist from center	Max dist from center	Average dist from center	Total clust count	Eps km	Min samples
  0	123456789_0	26.0941	80.1110	8630	0.861	0.061	3.347	1.82	884	0.5	50
  1	123456789_1	26.0898	80.1753	8630	5.964	0.002	0.981	0.02	577	0.5	50
  2	123456789_2	33.0215	117.0917	16010	34.966	0.008	0.039	0.02	235	0.5	50
  3	205700000_0	46.1062	122.9561	16890	3.099	0.005	0.603	0.05	1434	0.5	50
  4	205700000_1	49.2984	123.2356	18150	8.794	0.0407	1.348	0.12	1472	0.5	50

  Here is the roll-up that includes the summaries for both [.5,50] and [.5,100].  If the execute_dbscan and the analyze_dbscan functions had been run with more hyperparameters, there would be a row for each set of hyperparameters that produced any clusters in the table below.
  params	eps_km	min_samples	numb_clusters	average_cluster_count	average_nearest_port_from_center	average_dist_from_center	average_max_dist_from_center	f_measure	precision	recall
  0.5_50	0.5	50	6613	1116.533	17.153	1.032	7.899	0.675	0.632	0.725
  0.5_100	0.5	100	5324	1350.522	11.887	0.367	5.778	0.688	0.681	0.694



  Run the “execute_dbscan” function with a wide-range of hyperparameters to cluster the clusters from the last round to super clusters.
  We are now entering the second round of clustering.  We will take the output of the first round of clustering, and cluster the centerpoint of each cluster to make superclusters. The source table for this process is each one of the summary tables produced by each set of hyperparameters run against the entire dataset.  Since we will run this clustering approach with a number of hyperparameters to identify the ideal supercluster parameters, the total number of runs is equal to (sets of hyperparameters in first round * sets of hyperparameters in second round).  Since we only used two sets in the first round, we can use a larger set in the second.

  Since our goal in the first round was to find the minimum parameters to create a reasonably accurate cluster, we likely have multiple clusters at or near the same port.  The goal in the second round is to cluster these smaller clusters together into—preferably one— smaller number of clusters.  For the actualy hyperparameters, we will use  [.5, 1, 2, 3, 5] for epsilons_km and [2, 3, 4, 5, 7, 10] for minimum samples.  With 30 sets of hyperparameters in round 2 and two set from round 1, we will explore at least 60 different hyperparameters.

  Run the “analyze_dbscan” function against the results from the second round of clustering.

  The resulting table is displayed below, but this function will reduce the actual output of the clustering approach to metrics and stats to assist analysis.  

  The fact that there is such a steep drop-off in recall when the parameters increase beyond .5km with a minimum sample of 2 and 3 suggests that there is a not insignificant number of ports that have very little activity and that the noise filter might need to be increased.

  params	eps_km	min_samples	numb_clusters	average_cluster_count	average_nearest_port_from_center	average_dist_from_center	average_max_dist_from_center	f_measure	precision	recall
  0.5_2	0.5	2	525	10.12	13.363	0.771	1.873	0.737	0.759	0.717
  0.5_3	0.5	3	358	13.908	10.218	0.943	2.56	0.66	0.77	0.578
  0.5_4	0.5	4	276	17.04	9.794	0.976	2.954	0.617	0.78	0.511
  0.5_5	0.5	5	226	19.796	9.659	1.108	3.48	0.595	0.789	0.478
  0.5_7	0.5	7	155	25.845	8.286	1.401	4.593	0.491	0.765	0.361
  0.5_10	0.5	10	109	32.807	8.301	1.779	6.071	0.407	0.803	0.272
  1.0_2	1.0	2	407	13.919	14.837	1.225	3.341	0.713	0.715	0.711
  1.0_3	1.0	3	274	19.704	11.415	1.493	4.635	0.656	0.733	0.594
  1.0_4	1.0	4	222	23.491	10.966	1.529	5.251	0.621	0.754	0.528
  1.0_5	1.0	5	193	26.207	9.845	1.441	5.627	0.592	0.763	0.483
  1.0_7	1.0	7	152	31.125	9.45	1.678	6.847	0.513	0.753	0.389
  1.0_10	1.0	10	112	38.652	9.86	2.208	9.175	0.435	0.753	0.306
  2.0_2	2.0	2	341	17.484	19.59	2.208	4.753	0.645	0.634	0.656
  2.0_3	2.0	3	219	26.11	13.768	2.179	6.143	0.602	0.683	0.539
  2.0_4	2.0	4	179	31.184	11.89	2.195	6.81	0.584	0.703	0.5
  2.0_5	2.0	5	145	37.428	9.56	2.028	7.359	0.566	0.718	0.467
  2.0_7	2.0	7	117	44.752	9.482	2.217	8.578	0.509	0.737	0.389
  2.0_10	2.0	10	95	52.463	9.236	2.444	11.167	0.428	0.714	0.306
  3.0_2	3.0	2	343	17.98	24.716	2.976	5.998	0.602	0.583	0.622
  3.0_3	3.0	3	219	27.027	16.456	3.115	7.849	0.554	0.605	0.511
  3.0_4	3.0	4	171	33.673	12.396	2.616	8.015	0.539	0.648	0.461
  3.0_5	3.0	5	147	38.395	11.486	2.654	8.595	0.523	0.661	0.433
  3.0_7	3.0	7	110	49.091	9.572	2.774	9.799	0.482	0.684	0.372
  3.0_10	3.0	10	89	58.393	9.867	2.895	10.945	0.412	0.688	0.294
  5.0_2	5.0	2	282	22.514	26.161	4.208	12.288	0.508	0.505	0.511
  5.0_3	5.0	3	201	30.781	21.103	5.046	16.383	0.459	0.503	0.422
  5.0_4	5.0	4	158	38.184	15.67	4.5	18.273	0.433	0.528	0.367
  5.0_5	5.0	5	140	42.371	15.188	4.669	19.997	0.434	0.557	0.356
  5.0_7	5.0	7	112	51.125	12.74	4.422	22.067	0.416	0.586	0.322
  5.0_10	5.0	10	86	63.663	11.129	4.623	25.504	0.346	0.562	0.25




# Network Analysis
  First step is to create the input for a network multigraph.  For each unique identifier, lets evaluate if each point is "in" a port, as defined as a certain distance from a known port.  Then we can reduce all of the points down to when each unique identifier arrives and departs a known port.  In this network, each node is a port, and the edges are the travels of one identifier from a port to another.

  The python script "port_activity" identifies all positions within a certain distance of a port (now set for 2000m).  It then finds the closest port if more than one is returned, and joins the closes port back to the position data as a new table.  Right now this is one large query and needs to be modified to insert the new columns for port_id and port_name back into the original data rather than make a new table.

  The Python script "network_building" iterates through a table with ship position and determines an origin and destination for each connection between two ports.  These can be imported into networkx as edges.  The script also captures departure time,  arrival time, and total positions between the two ports as edge attributes.  These can be used to narrow down true port-to-port trips and minimize times when a ship repeatedly jumps back and forth between ports in a short number of positions or narrow time window.  All of this data is written to a table in the database.

### Status as of 18 January 2019:
  Using a sample of 200 mmsis, we went from 135 million positions in all of January to a total of 2,155,696 positions.  This reduces to 1003 nodes.


  - I also need to add a conditional that looks for a minimum of X time at each port to prevent a ship from traveling by numerous ports to be listed as in port.
  - Refactor not to use pandas if proved to be non-preformant
  - Break code block into functions

### Note on Network Building
So options.
 - Create a knn script that finds the nearest port for every position report.  Done. Problem is that the network script needs the postions further than xkm and within xkm.
 - Find nearest ports with 5, 10, xkm from every position.

# Time series analysis
  Holt-winter seasonal model, multiple linear regression, ARMA, ARIMA, SARIMAX, and maybe even LSTM?  Predict volume at a port or for ports?  Or predict activity for a class of ships?
