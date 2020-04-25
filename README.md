# AIS_project Overview

This project's goals are to:
- Gain experience building out data pipelines for large, geospatial datasets myself
- Learn more about PostGIS and QGIS
- Practice good software engineering practices in my data science projects
- Experiment with different ways of identifying and evaluating clusters in space and time
- Developing optimal implementations of oft-used functions, such as finding distances between two lists
- Translating dense spatial-temporal data into networks
- Analyze network analysis, including machine learning for prediction

The project has four major phases.
  1. Data ingest, cleaning, and analysis.
  2. Cluster creation and evaluation.
  3. Network analysis and prediction.
  4. Time series analysis and prediction.

  Each phase has multiple individual projects.  This readme will try to document the project and track development.

  ## Data Sources
  Easy to scrape website
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
  - psycopg2 (trying to replace all instances with sqlalchemy)
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

  # Data Ingest, Cleaning, and Analysis

  The AIS data is large.  January 2017 data fro the US is 25 gigabytes.  The entire year could be about 300 gigabytes.  There are two options here.  The first is to put it all in the cloud with all the raw data and cleaned tables.  The second is to  store all csvs as zipped files, process and clean he data in chunks to a database, and create a summary for every vessel in the data.  Then, we can sample the raw positions and conduct cluster analysis on those samples.  Armed with our summary analysis of the entire data, we can then ensure that our samples are representative of different patterns.

  We will first pursue the latter option, but may eventually use AWS to spin-up a PostGres instance.  

  Current implementation (16 January 2020) uses a source directory as an input and iterates through all the csv files in that directory.  Future improvements can allow filtering based on month and zone.  Additionally, can eventually add a webscraper component.

  Still to do on ingest script:
  - Set up scrape from remote website
  - set up automatic filters for zones and months so they dont have to be manually selected.

  ### Ingest pipelines
  First we create a new database (in prod it is the "AIS_data" database) and then build a Post GIS extension.  Then we run the python script "ingest_script_prod".  This script:

  - Creates a connection to the db
  - Drops any existing tables using a custom 'drop_table' function.  (Note, in prod the actual calls are commented out for safety.)
  - Creates a "dedupe_table" function for the ship_info table.
  - Creates the tables 'imported_ais' to hold the original copied data, the ship_info table that includes mmsi, ship name and ship type, the ship_position table with location data for each position, and the wpi table that has info from the world port index for each major port.
  - The "parse_ais_SQL" function ingests the raw csv file to imported_ais and then selects the relevant data for the ship_info and ship_position table.
  - The "make_ship_trips" function creates a ship_trips table that reduces each ship's activity in the ship_position table to summary stats, including a line of the ship's voyage.
  - All activities are recorded to a processing log ("proc_log") with the date appended.  "function_tracker" function executes key functions, logs times, and prints updates to the console.

  ### Reducing Raw position data
  Unfortunately analysis against the entire corpus is difficult with limited compute.   However, we can reduce the raw positions of each ship into a line segment that connects each point together in the order of time.  Because there are sometimes gaps in coverage, a line will not follow the actual ships path and "jump".  The "make_ship_trips" function in the ingest pipeline uses geometry in PostGIS to connect all positions per ship, but this is not accurate as the coverage is not consistent.

  ### Analyze Summarized Ship trips
  Since we don't want to test all of our algorithm development against the entirety of the data, we need to select a sample of MMSI from the entire population to examine further.  The Jupyter Notebook "ships_trips_analysis" parses the ships_trips table and analyzes the fields.  We can use it to select a sample of MMSIs for further analysis.  The notebook builds several graphs and examines the summary of each MMSI's activity.  After filtering out MMSIs that don't travel far, have very few events, or travel around the globe in less than a month and are possible errors, the notebook exports a sample (250 now) to a csv file.  The notebook also writes the ship_trips sample to a new table, "ship_trips_sample" directly from the notebook using SQL Alchemy.

  ### Create Samples for Further analysis
  Python script "populate_sample" takes the csv output of the sample MMSIs from the Jupyter Notebook "ships_trips_analysis" and adds all positions from the "ship_position" table to a new "ship_position_sample" table.  It also makes a "ship_trips_sample" table from the full "ship_trips" table.

  ### Port Activity tables creation
  Creates a table that includes all of a ship's position when the ship's positions are within X meters of a known port.  This table has all of these positions labeled with that port. Then

  The WPI dataset has  duplicate port locations.  Specifically, the exact same geos are used twice by two different named and indexed ports 13 times.  I naively dropped duplicates on the lat and lon columns to resolve.  No order is guaranteed, so this is a risk for reproducibility.  TODO: EIther order the drops in some way or record the 13 ports removed for documentation.

  Still TODO
  - rejoin this back to the main tables since we want to have a port for each position as well as the blanks.  Making a new table is fine for the sample data, but not for the full data.

  ### Table Summary for ingest pipeline
  imported_ais --> ship_position --> ship_trips --> port_activity
  TODO add ERD here.

  ### Lessons Learned
  #### Using PostGreSQL COPY
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

  ## Clustering Problem Description
  Unsupervised clustering is regularly explained as a key field in machine learning, but is often an after-thought in a world of easy-to-build, high-powered neural networks.  However, I argue there are many viable use cases for straight-forward but effective unsupervised clustering approaches, particularly in areas where these is a large amount of data and only a small portion can be accurately labeled manually.  

  I argue that one of these use cases is dense, geospatial data where we may only care about activity near certain areas.  Often we know where some many of those areas are, but it’s difficult for a human or a machine to find new, similar areas in dense geospatial data.  If we can build an unsupervised clustering methodology that identifies the known locations with a high level of accuracy, we can assume that the same methodology with the same hyperparameters can find other, similar areas.  

  A major limitation with unsupervised clustering is evaluating how well a clustering approach performed in grouping “like with like.”  Evaluating decision boundaries and comparing means and variances across groups can give a good idea how precise an unsupervised clustering algorithm separated observations, and have been used in many cases to get impactful, real-world results.  

  However, when using unsupervised clustering to group observations into groups that exist in the real world, it’s difficult to compare these groups to unlabeled data.  Since unsupervised clustering approaches like K-means, hierarchical clustering, and DBSCAN will cluster observations together in groups that are numbered as they are identified, how can you tell if these groups correspond to the real-world groupings you are trying to achieve? For a concrete example with our AIS data, how can we tell if a model’s results identified ships’ positions at a port, or if the cluster is in the middle of the ocean?  Individual clusters can be manually reviewed, but how can one execute this at scale, with millions of points and thousands or tens of thousands of clusters?

  To accomplish this, we need an effective approach to evaluate an unsupervised clustering model’s accuracy across many different hyperparameters.  I propose a new methodology that will set three different accuracy metrics for each unsupervised clustering model and then learn the hyperparameters that minimize the errors across these metrics.  

  ## TODO
  Note, when we get there, we will have to look at how to represent time.  If we include it as a Z dimension, the scale will impact different clustering spaces.  Perhaps we can run the same clustering approaches with different time scales to show how it can impact the final clusters.  Or we could just ignore time entirely and cluster just based on spatial activity.

  Can I use labeled clusters of "ports" to identify the critical values for distance in time and space, and then apply those parameters against the rest of the data.
  -Likely DBSCAN is the best implementation
  -Would Gaussian mixture models successfully identify anamalous ship traffic?

  So use DBSCAN to cluster individual ships, but it crashes local machine in sklearn.  Rebuilt code to work in PostGres and can now run locally or in the cloud.  We can run the clustering for each ship, and then cluster all the ports from all the ships to find final ports.  This may be less subject to drift over time.  Our hyperparametrs are only good for the volume of ships in the timeframe.  Say we had well-tuned parameters for a month of AIS data.  They would be far too low for a years worth of data, because 12 times the shipping activity is in the same area.  If we cluster the activity first and then cluster to find ports, its could be more resistent to walking.

  Need to do:
  - on the purity analysis, need to compare when closest port == most strongly repersented port.
  - filter out points that are far from known ports (activity on mississippi river)
  - add column for composition, ie how many unique mmsis are in each cluster.  penalize singletons.
  - cluster max width



  # Network Analysis

  First step is to create the input for a network multigraph.  For each unique identifier, lets evaluate if each point is "in" a port, as defined as a certain distance from a known port.  Then we can reduce all of the points down to when each unique identifier arrives and departs a known port.  In this network, each node is a port, and the edges are the travels of one identifier from a port to another.

  The python script "port_activity" identifies all positions within a certain distance of a port (now set for 2000m).  It then finds the closest port if more than one is returned, and joins the closes port back to the position data as a new table.  Right now this is one large query and needs to be modified to insert the new columns for port_id and port_name back into the original data rather than make a new table.

  The Python script "network_building" iterates through a table with ship position and determines an origin and destination for each connection between two ports.  These can be imported into networkx as edges.  The script also captures departure time,  arrival time, and total positions between the two ports as edge attributes.  These can be used to narrow down true port-to-port trips and minimize times when a ship repeatedly jumps back and forth between ports in a short number of positions or narrow time window.  All of this data is written to a table in the database.

  ### Status as of 18 January 2019:
  Using a sample of 200 mmsis, we went from 135 million positions in all of January to a total of 2,155,696 positions.  This reduces to 1003 nodes.


  - I also need to add a conditional that looks for a minimum of X time at each port to prevent a ship from traveling by numerous ports to be listed as in port.
  - Refactor not to use pandas if proved to be non-preformant
  - Break code block into functions

  # Time series analysis
  Holt-winter seasonal model, multiple linear regression, ARMA, ARIMA, SARIMAX, and maybe even LSTM?  Predict volume at a port or for ports?  Or predict activity for a class of ships?
