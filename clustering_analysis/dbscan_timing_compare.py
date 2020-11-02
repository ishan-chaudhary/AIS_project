import pandas as pd
import numpy as np
import folium
import datetime
from gnact import dbscan
from gnact import stdbscan

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

from gnact import utils, clust
import gsta_config

import warnings
warnings.filterwarnings('ignore')
#%%
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

#%%
df_results = clust.get_clusters(df_posits, eps_km=3, min_samp=100, method='dbscan')

#%%
# need new unique cluster ids across each uid.
clust_count = 0
# will hold results of second round temporal clustering
df_second_round = pd.DataFrame()

# begin iteration.  Look at each cluster in turn from first round results
clusters = df_results['clust_id'].unique()
for c in clusters:
    df_c = df_results[df_results['clust_id'] == c]
    X = ((df_c['time'].astype('int').values) / ((10**9)*60)).reshape(-1,1)
    x_id = df_c.loc[:, 'id'].astype('int').values
    # cluster again using DBSCAN with a temportal epsilon (minutes) in one dimension
    dbscan = DBSCAN(eps=600, min_samples=100, algorithm='kd_tree',
                    metric='euclidean', n_jobs=1)
    dbscan.fit(X)
    results2_dict = {'id': x_id, 'clust_id': dbscan.labels_}
    # gather the output as a dataframe
    df_results2 = pd.DataFrame(results2_dict)
    df_results2 = df_results2[df_results2['clust_id'] != -1]
    clusters2 = df_results2['clust_id'].unique()
    for c2 in clusters2:
        df_c2 = df_results2[df_results2['clust_id'] == int(c2)] # need int rather than numpy.int64
        # need to assign a new cluster id
        df_c2['clust_id'] = clust_count
        df_second_round = df_second_round.append(df_c2)
        clust_count +=1

df_second_results = pd.merge(df_second_round, df_results.drop('clust_id', axis=1), how='left', left_on='id', right_on='id')
df_centers = clust.calc_centers(df_second_results)

#%%
df_stdbscan = pd.read_csv('st_dbscan_results.csv', parse_dates=['time'])
df_stdbscan = df_stdbscan[df_stdbscan['clust_id']!=-1]
df_stdbscan_centers = clust.calc_centers(df_results)

