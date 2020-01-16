# AIS_project

This project's goals are to:
- Gain experience building out data pipelines myself
- Learn more about PostGIS and QGIS
- Better practice good software engineering practices in my data science projects
- Experiment with different ways of identifying and evaluating clusters in space and time
- Translating dense spatial-temporal data into networks
- Analyze network analysis, including machine learning for prediction

The project has three major phases.
  1. Data ingest, cleaning, and analysis.
  2. Cluster creation and evaluation.
  3. Network analysis and prediction.

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

  ## Data Ingest, Cleaning, and Analysis

  The AIS data is large.  January 2017 data fro the US is 25 gigabytes.  The entire year could be about 300 gigabytes.  There are two options here.  The first is to put it all in the cloud with all the raw data and cleaned tables.  The second is to  store all csvs as zipped files, process and clean he data in chunks to a database, and create a summary for every vessel in the data.  Then, we can sample the raw positions and conduct cluster analysis on those samples.  Armed with our summary analysis of the entire data, we can then ensure that our samples are repersentative of different patterns.

  We will first pursue the latter option, but may eventually use AWS to spin-up a PostGres instance.  

  Current implementation (16 January 2019) uses a source directory as an input and iterates through all the csv files in that directory.  Future improvements can allow filtering based on month and zone.  Additionally, can eventually add a webscraper component.


  Still to do on ingest script:
  - Set up scrape from remote website
  - set up filters for zones and months

  ### Ingest pipelines
  First we create a new database (in prod it is the "AIS_data" database)m and then build a Post GIS extension.  Then we run the python script "ingest_script_prod".  This script:

  - Creates a connection to the db
  - Drops any existing tables using a custom 'drop_table' function.  (Note, in prod the actual calls are commented out for safety.)
  - Creates a "dedupe_table" function for the ship_info table.
  - Creates the tables 'imported_ais' to hold the original copied data, the ship_info table that includes mmsi, ship name and ship type, the ship_position table with location data for each position, and the wpi table that has info from the world port index for each major port.
  - The "parse_ais_SQL" function ingests the raw csv file to imported_ais and then selects the relevant data for the ship_info and ship_position table.
  - The "make_ship_trips" function creates a ship_trips table that reduces each ship's activity in the ship_position table to summary stats, including a line of the ship's voyage.
  - All activities are recorded to a processing log ("proc_log") with the date appended.  "function_tracker" function executes key functions, logs times, and prints updates to the console.

  ### Reducing Raw position data
  Unfortunately we have to use the more dense point data rather than the lines.  Because there are sometimes gaps in coverage, a line will not follow the actual ships path and "jump".  If the line is near any port, it would trigger a false positive.  The "make_ship_trips" function in the ingest pipeline uses geometry in PostGIS to connect all positions per ship, but this is not accurate as the coverage is not consistent.

  ### Analyze Summarized Ship trips
  Since we don't want to test all of our algorithm development against the entirety of the data, we need to select a sample of MMSI from the entire population to examine further.  The Jupyter Notebook "ships_trips_analysis" parses the ships_trips table and analyzes the fields.  We can use it to select a sample of MMSIs for further analysis.  The notebook builds several graphs and examines the summary of each MMSI's activity.  After filtering out MMSIs that don't travel far, have very few events, or travel around the globe in less than a month and are possible errors, the notebook exports a sample (250 now) to a csv file.  The notebook also writes the ship_trips sample to a new table, "ship_trips_sample" directly from the notebook using SQL Alchemy.

  ### Create Samples for Further analysis
  Python script "populate_sample" takes the csv output of the sample MMSIs from the Jupyter Notebook "ships_trips_analysis" and adds all positions from the "ship_position" table to a new "ship_position_sample" table.  It also makes a "ship_trips_sample" table from the full "ship_trips" table.

  ### Lessons Learned
  #### Using PostGreSQL COPY rather than iterating through chunks using pandas
  Using Pandas and iterating through 100000 rows at a time on a sample csv of 150 mb took ~2 mins.  By using copy to create a temp table and then selecting the relevant info to populate the ship_info and ship_position table, the total time was reduced to 25 seconds.

  ### #A note on Spatial Indices
  I first failed to create spatial indices at all.  This led to a 2 minute 45 second spatial join between a table with 9265 positions and the WPI table with 3630 ports.  By adding a spatial index with the below syntax to both tables, the query then took only 124 msec.  Running the same query against over 2 million rows in the sample data took 2.6 seconds.

  CREATE INDEX wpi_geog_idx
  ON wpi
  USING GIST (geog);
  CREATE INDEX ship_test_geog_idx
  ON ship_test
  USING GIST (geog);

  ## Notes on Visualizing in QGIS

  Large numbers of points within a vector layer can severely impact rendering performance within QGIS.  Recommend using the "Set Layer Scale Visibility" to only show the points at a certain scale.  Currently I found a minimum of 1:100000 appropriate with no maximum.  Ship trips can have no scale visibility because they render much faster.


  ## Clustering

  Note, when we get there, we will have to look at how to represent time.  If we include it as a Z dimension, the scale will impact different clustering spaces.  Perhaps we can run the same clustering approaches with different time scales to show how it can impact the final clusters.

  Can I use labeled clusters of "ports" to identify the critical values for distance in time and space, and then apply those parameters against the rest of the data.

  ## Network Analysis

  First step is to create the input for a network multigraph.  For each unique identifier, lets evaluate if each point is "in" a port, as defined as a certain distance from a known port.  Then we can reduce all of the points down to when each unique identifier arrives and departs a known port.  In this network, each node is a port, and the edges are the travels of one identifier from a port to another.  


  - I also need to add a conditional that looks for a minimum of X time at each port to prevent a ship from traveling by numerous ports to be listed as in port.
  - Also need to catch ships departing a port and then returning to the same.  Right now that is missed and treated all as one port visit.
