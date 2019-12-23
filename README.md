# AIS_project

This project's goals are to:
- Gain experience building out data pipelines myself
- Learn more about PostGIS and QGIS
- Better practice good software engineering practices in my data science projects
- Evaluate different ways of identifying and evaluating clusters in space and time
- Translating dense spatial-temporal data into networks
- Analyze network analysis, including machine learning for prediction

The project has three major phases.
  1. Data ingest, cleaning, and analysis.
  2. Cluster creation and evaluation.
  3. Network analysis and prediction.
  
  Each phase has multiple individual projects.  This readme will try to document the project and track development.
  
  ## Required Software
  - PostGreSQL with post GIS
  - PG Admin 4
  - QGIS
  - Spyder
  
  ## Data Ingest, Cleaning, and Analysis
  
  The AIS data is large.  January 2017 data fro the US is 25 gigabytes.  The entire year could be about 300 gigabytes.  There are two options here.  The first is to put it all in the cloud with all the raw data and cleaned tables.  The second is to  store all csvs as zipped files, process tand clean he data in chunks to a database, and create a summary for every vessel in the data.  Then, we can sample the raw positions and conduct cluster analysis on those samples.  Armed with our summary analysis of the entire data, we can then ensure that our samples are repersentative of different patterns.
  
  We will first pursue the latter option, but may eventually use AWS to spin-up a PostGres instance.
  
  Tasks:
  - Iterate through all the raw files, and zip them after I download them.
  - Wrtie a function to iterate through all the files, unzip them, open them in chunks, process them, close the file, and rezip it.
  - Write a SQL command to summarize the millions of ship positions and times to one row per ship.  Summary information will include a line segment of the ship's trip, that length, the number of geolocation, first and last date detected in the data, and total time active in the data.
  - Analyze this summary data and select N number of ships to explore clustering approaches on.
  - Create a new table (or database) to use for clustering.
