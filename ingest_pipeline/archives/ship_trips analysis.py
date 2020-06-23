#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import datetime
import random
get_ipython().run_line_magic('matplotlib', 'inline')


# The goal of this notebook is to analyze all available information about ships to identify common trends and select a sample repersentative of the whole. To achieve this, we will make use of the "ship_trips" table, which reduces all individual ship positions to a single line, or trip.  From this line, we can determine the total distance traveled by a ship.  From the raw data, we can also determine the total number of days the ship is active in the data and the total count of position reports.
# 
# There are of course tricky aspects of this data.  Some AIS transponders are stationary, and should not be considered in this dataset.  Other transponders could be moved from ship to ship.  Finally, our coverage is not universal.  We are using Coast Guard data acquired from terrestial, coastal sites.  These reduce our AIS coverage to within about 30-50 miles of the US coastline.  Not only does this only give us a limited picture of the activity, but it leads to ships jumping from one coast to another when they leave the coastal coverage area.

# First we will create a database connection using psycopg2 and read in all the data from the ship_trips table.

# In[8]:


conn = psycopg2.connect(host="localhost",database="ais_test")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()


# In[9]:


# select all the summary data from ship_trips
c = conn.cursor()
df_trips = pd.read_sql('select * from ship_trips', conn)
c.close()


# ## EDA of ship_trips info

# In[10]:


df_trips.info()


# In[11]:


df_trips.describe()


# From this plot below, we can see that many ships have a relatively low number of position reports, but there are some with over 40,000 positions in a one month timeframe.  That's almost one position every minute of every day!

# In[12]:


sns.distplot(df_trips['position_count'])


# The distirbution of the ships' length of travel in days is definitely bimodal.  Many ships are only active one or two days, while others are active every day in the reporting period.  It's likely that some ships on the low end of the distribution had a handful of positions while transiting coverage areas, or were only active in AIS for a short period of time.  Also, because we are looking at only a month, its possible a ship was in port for the majority of January until the very end of the month.

# In[13]:


df_trips['trip_days'] = df_trips['time_diff'].dt.days
sns.distplot(df_trips['trip_days'])


# Regarding total voayage, a few outliers have really scaled out this plot.  We'll have to look at this a little more closely.

# In[14]:


sns.distplot(df_trips['line_length_km'])


# In[15]:


g = sns.pairplot(df_trips, vars=['position_count', 'trip_days', 'line_length_km'])


# ## Taking a Sample of ship_trips

# From the data we have already seen, we can filter out ships from our data that have abnormally long or short travel lengths, or ships who are only in the data for less than three days.  This reduces our total sample size from 20,731 by about half!

# In[19]:


df_filtered = df_trips[(df_trips['line_length_km'] < 30000) & # We want to avoid trips that might be errors
                       (df_trips['line_length_km'] > 100) & # avoid trips that are too short or stationary
                       (df_trips['time_diff'] > pd.Timedelta('3 days'))] # avoid trips of short duration
print(len(df_filtered))


# From the pairplot below, we can see there are still some outliers that have travel much further than other ships with the same number of positions and days underway.  These ships might be those that jump from one area of collection to another, and can be included to get an idea of how this anomoly will affect our algorithms.  Other than that, it looks like we are ready to go!

# In[18]:


g = sns.pairplot(df_filtered, vars=['position_count', 'trip_days', 'line_length_km'])


# In[33]:


# use df.sample to take 200 samples
df_sample = df_filtered.sample(n=200, replace=False, random_state=1)


# In[29]:


sns.distplot(df_sample['position_count'])


# In[22]:


df_sample['trip_days'] = df_sample['time_diff'].dt.days
sns.distplot(df_sample['trip_days'])


# In[23]:


sns.distplot(df_sample['line_length_km'])


# In[24]:


g = sns.pairplot(df_sample, vars=['position_count', 'trip_days', 'line_length_km'])


# In[25]:


# total count of positions here will help verify data integrity when ingested to the database.  
# incorporate in future unit testing.
df_sample['position_count'].sum()


# In[82]:


# write the samples to a csv
df_sample['mmsi'].to_frame().to_csv('sample_mmsi.csv', index=False)


# In[88]:


# create a SQL alchemy engine and write the samples to a new table in the database.
from sqlalchemy import create_engine
engine = create_engine('postgresql://patrickmaus:@localhost:5432/ais_data')

df_sample.to_sql('ship_trips_sample', engine, if_exists='fail', index=False)

c = conn.cursor()
c.execute("""ALTER TABLE ship_trips_sample ALTER COLUMN line TYPE Geometry;""")
conn.commit()
c.close()
 

