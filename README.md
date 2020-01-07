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

  Tasks:
  - Iterate through all the raw files, and zip them after I download them.
  - Write a function to iterate through all the files, unzip them, open them in chunks, process them, close the file, and rezip it.
  - Write a SQL command to summarize the millions of ship positions and times to one row per ship.  Summary information will include a line segment of the ship's trip, that length, the number of geolocation, first and last date detected in the data, and total time active in the data.
  - Analyze this summary data and select N number of ships to explore clustering approaches on.
  - Create a new table (or database) to use for clustering.

  Still to do on ingest script:
  - Set up scrape from remote website
  - set up filters for zones and months
  - ingest ship info table as well
  - move away from pandas with chunks

  ### Ingest pipelines
  First we create a new database "AIS_data" and then build a Post GIS extension.  Then we run the python script "ingest script".  This will establish a connection to the database, create a table "ship_info" and a table "ship_position".  We'll then read the AIS data from a source directory, unzip each file, and read each csv file in chunks using pandas.  We'll write data to "ship_info" and "ship_position".

  Still to do on ingest script:
  - Set up scrape from remote website
  - set up filters for zones and months
  - ingest ship info table as well
  - move away from pandas with chunks

  ### Analyze Bulk Ingested Data
  The next step is to summarize the millions of ship positions to a smaller table with each MMSI, the total number of positions for each MMSI, a geometry object that reduces the points to a line, the length of that line in kilometers, the first date a MMSI was "heard", the last date the MMSI was "heard", and the total length of time between first and last "heard".  We can use the SQL script "summarize_ship_trips" to create this "ship_trip" table.


  ### Analyze Summarized Ship trips
  Since we dont want to test all of our algorithm development against the entirety of the data, we need to select a sample of MMSI from the entire population to examine further.  The Jupyter Notebook "ships_trips_analysis" parses the ships_trips table and analyzes the fields.  We can use it to select a sample of MMSIs for further analysis.  The notebook builds several graphs and examines the summary of each MMSI's activity.  After filtering out MMSIs that dont travel far, have very few events, or travel around the globe in less than a month and are possible errors, the notebook exports a sample (250 now) to a csv file.  The notebook also writes the ship_trips sample to a new table, "ship_trips_sample" directly from the notebook using SQL Alchemy.

  ### Create Samples for Further analysis
  Python script "populate_sample" takes the csv output of the sample MMSIs from the Jupyter Notebook "ships_trips_analysis" and adds all positions from the "ship_position" table to a new "ship_position_sample" table.

  ## Notes on Visualizing in QGIS

  Large numbers of points within a vector layer can severely impact rendering performance within QGIS.  Recommend using the "Set Layer Scale Visibility" to only show the points at a certain scale.  Currently I found a minimum of 1:100000 appropriate with no maximum.  Ship trips can have no scale visbility because they render much faster.


  ## Clustering

  Note, when we get there, we will have to look at how to repersent time.  If we include it as a Z dimension, the scale will impact different clustering spaces.  Perhaps we can run the same clustering approaches with different time scales to show how it can impact the final clusters.

  Can I use labeled clusters of "ports" to identify the critical values for distance in time and space, and then apply those parameters against the rest of the data.

  ## Network Analysis

  First step is to create the input for a network multigraph.  For each unique identifier, lets evaluate if each point is "in" a port, as defined as a certain distance from a known port.  Then we can reduce all of the points down to when each unique identifier arrives and departs a known port.  In this network, each node is a port, and the edges are the travels of one identifier from a port to another.  
