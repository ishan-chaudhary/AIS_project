# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

import datetime
import numpy as np

import random
from collections import defaultdict
from nltk import ngrams
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Process
from multiprocessing import Pool
#%%
# establish connections and get the edgelist from the database.
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_full_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_full_params)

# df_edgelist is already sorted by time using the gsta.get_edgelist function
df_edgelist = gsta.get_edgelist(edge_table='cargo_edgelist', engine=loc_engine, loiter_time=2)
df_edgelist.to_csv('full_edgelist_for_nlp.csv')

df_edgelist = df_edgelist[['Source', 'Target', 'uid']]
# Source and Target should be capitilzed so the network can be read into Gephi.
df_edgelist.columns = ['Source', 'Target', 'uid']

#%%
sep = pd.read_csv('air_data/sep_2019.csv')
oct = pd.read_csv('air_data/oct_2019.csv')
nov = pd.read_csv('air_data/nov_2019.csv')
dec = pd.read_csv('air_data/dec_2019.csv')
jan = pd.read_csv('air_data/jan_2020.csv')
feb = pd.read_csv('air_data/feb_2020.csv')

# put together the month pieces
df_flights = pd.concat([sep, oct, nov, dec, jan, feb])
del (sep, oct, nov, dec, jan, feb)

# need to sort so rows are sequential
df_flights = df_flights.sort_values(['TAIL_NUM', 'FL_DATE', 'DEP_TIME'])
#%%
df_edgelist = pd.concat([df_flights['ORIGIN'], df_flights['DEST'], df_flights['TAIL_NUM']], axis=1)
# Source and Target should be capitilzed so the network can be read into Gephi.
df_edgelist.columns = ['Source', 'Target', 'uid']
df_edgelist.dropna(inplace=True)




# %% build history and split to test/train with 20% test
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

history_orig = gsta.build_history_multiprocess(df_edgelist)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)
