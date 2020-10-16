import pandas as pd
import numpy as np
import folium
import datetime
from gnact import dbscan
from gnact import stdbscan

from gnact import utils, clust
import gsta_config

import warnings
warnings.filterwarnings('ignore')

# create the engine to the database
engine = utils.connect_engine(gsta_config.colone_cargo_params, print_verbose=True)
# make the df from the data in the database for MSC Ashrui
df_posits = clust.get_uid_posits(('636016432',), engine, end_time='2018-01-01')

values = [10, 100, 200, 500, 1000, 1500, 2000, 5000, 10000]

#%%
st_times = []
db_times = []
for v in values:
    tick = datetime.datetime.now()
    df_results = stdbscan.ST_DBSCAN(df_posits[:v], spatial_threshold=3, temporal_threshold=600, min_neighbors=100)
    lapse = datetime.datetime.now() - tick
    print(v, lapse.total_seconds())
    st_times.append(lapse.total_seconds())

    tick = datetime.datetime.now()
    df_results = dbscan.DBSCAN(df_posits[:v], spatial_threshold=3, min_neighbors=100)
    lapse = datetime.datetime.now() - tick
    print(v, lapse.total_seconds())
    db_times.append(lapse.total_seconds())


#%%
sk_db_times = []
for v in values:
    tick = datetime.datetime.now()
    df_results = clust.get_clusters(df_posits[:v], eps_km=3, min_samp=100, method='dbscan')
    lapse = datetime.datetime.now() - tick
    print(v, lapse.total_seconds())
    sk_db_times.append(lapse.total_seconds())

#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(values, sk_db_times, label="clustering_analysis in Scikit-Learn")
ax.plot(values, st_times, label="ST_DBSCAN in Pandas")
ax.plot(values, db_times, label="clustering_analysis in Pandas")
plt.title('Time Complexity Comparisons for Different clustering_analysis Implementations')
plt.legend()
plt.ylabel('Time in Seconds')
plt.xlabel('Number of Points')
plt.show()