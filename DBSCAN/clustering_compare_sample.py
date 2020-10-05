# plotting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

# reload modules when making edits
from importlib import reload

reload(gsta)

#%%
df = pd.read_csv('sample_ship_posits.csv')
df['time'] = pd.to_datetime(df['time'])
try:
    df.drop(['Unnamed: 0'], inplace=True, axis=1)
except Exception as e:
    print (e)
# %%
# conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)

# %% create a df from the database
df = pd.read_sql_query("SELECT id, time, lat, lon, cog, sog, status, anchored, moored, underway"
                       " FROM ais_cargo.public.uid_positions "
                       "WHERE uid = '636016432'"
                       "ORDER BY time", loc_engine)
df.to_csv('sample_ship_posits.csv', index=False)

#%%
def calc_dist(df, unit='nm'):
    """
    Takes a df with id, lat, and lon and returns the distance  between
    the previous point to the current point as a series.
    :param df:
    :return:
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [df.lon.shift(1), df.lat.shift(1), df.lon, df.lat])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    r = 2 * np.arcsin(np.sqrt(a))
    if unit =='mile':
        return 3958.748 * r
    if unit =='km':
        return 6371 * r
    if unit =='nm':
        return 3440.65 * r
    else: print("Unit is not valid.  Please use 'mile', 'km', or 'nm'.")
        return None


def calc_bearing(df):
    """
    Takes a df with id, lat, and lon and returns the computed bearing between
    the previous point to the current point as a series.
    :param df:
    :return:
    """
    lat1 = np.radians(df.lat.shift(1))
    lat2 = np.radians(df.lat)
    dlon = np.radians(df.lon - df.lon.shift(1))
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)
    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return round(compass_bearing, 2)

def traj_enhance_df(df):
    """
    Takes a df with id, lat, lon, and time.  Returns a df with these same columns as well as
    time_rounded to the lowest minute, time difference from current point and previous group,
    time_diff_hours, course over ground, distance traveled since last point, and speed in knots
    :param df:
    :return:
    """
    # we want to round by minute and drop any duplicates
    df['time_rounded'] = df.time.apply(lambda x: x.floor('min'))
    df.drop_duplicates(['time'], keep='first', inplace=True)
    # calculate time diff between two points
    df['time_diff'] = df.time - df.time.shift(1)
    # time diff in hours needed for speed calc
    df['time_diff_hours'] = pd.to_timedelta(df.time_diff, errors='coerce').dt.total_seconds() / 3600
    # calculate bearing, distance, and speed
    df['cog'] = calc_bearing(df)
    df['dist_nm'] = calc_dist(df, unit='nm')
    df['speed_kts'] = df['dist_nm'] / df['time_diff_hours']
    return df

#%%
df_enhance = enhance_df(df)

#%%
df_enhance[df_enhance['speed_kts']<35].boxplot(column='speed_kts', by='status')
plt.show()
#%%
fig, ax = plt.subplots()
plt.()



#%%


df_results =clust.get_clusters(df_enhance[df_enhance['speed_kts']<2], eps_km=3, min_samp=200, method='dbscan')



# %% feature engineering
# if we want to round by minute
df['time_rounded'] = df.time.apply(lambda x: x.floor('min'))
# after rounding by minute, drop any duplicates
df = df.drop_duplicates(['time'], keep='first')

# transferring datetime to int gives nanonseconds since time zero.
# this transformation will give you time since zero IN MINUTES.  Needed for ST OPTICS or DBSCAN.
df['ts_minutes'] = df.time.astype('int') / 60 / (10 ** 9)

# calculate time diff between two points
df['time_diff'] = df.time - df.time.shift(1)
# time diff in hours needed for speed calc
df['time_diff_hours'] = pd.to_timedelta(df.time_diff, errors='coerce').dt.total_seconds() / 3600

df['c_cog'] = calc_bearing(df)
df['dist_nm'] = calc_dist(df, unit='nm')
df['speed_kts'] = df['dist_nm']/df['time_diff_hours']

#%% write to sql
df.to_sql(name='ship_sample_imp', con=loc_engine, if_exists='replace', method='multi', index=False)
#%%
df.to_csv('ship_sample_imporved.csv')
#%%
stopped = df[df['underway']!=True]
underway = df[df['underway']==True]

#%%
stopped.speed_knts.mean()
#%%
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())
from sklearn.cluster import OPTICS
min_samp = 250

# format data for dbscan
X = (np.radians(df.loc[:, ['lon', 'lat']].values))
x_id = df.loc[:, 'id'].values

# execute sklearn's OPTICS
# 5km in radians is max eps
optics = OPTICS(min_samples=min_samp, max_eps=5 / 6371.0088, metric='euclidean', cluster_method='xi',
                algorithm='kd_tree', n_jobs=1)
optics.fit(X)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)


#%%
results_dict = {'id': x_id, 'clust_id': optics.labels_, 'lat': df.lat, 'lon': df.lon}
df_results = pd.DataFrame(results_dict)
# drop all -1 clust_id, which are all points not in clusters
df_results = df_results[df_results['clust_id'] != -1]


#%%
df_centers = gsta.calc_centers(df_results)
df_centers.to_csv('DBSCAN/sample_centers.csv', index=False)

#%%
from importlib import reload
import gnact
reload(gnact)
from gnact import clust
from gnact import utils
import pandas as pd
import numpy as np
import gsta_config
#%%
engine = gnact.utils.connect_engine(gsta_config.colone_cargo_params, print_verbose=False)

df = gnact.clust.get_uid_posits(('636016432',), engine, end_time='2018-01-01')
df_results =clust.get_clusters(df, eps_km=3, min_samp=200, time_window=0, method='dbscan')
#%%
df_results =clust.get_clusters(df, eps_km=3, min_samp=200, time_window=0, method='dbscan')
#%%
import sys
import math
sample = X[:100]
from scipy.spatial.distance import pdist, squareform
n, m = sample.shape
time_dist = squareform(pdist(sample[:, 0].reshape(n, 1),
                             metric='euclidean'))
time_filter = math.pow(10, m)
print(sys.getsizeof(time_dist)/1000000)
print(sys.getsizeof(sample)/1000000)
#%%
euc_dist = squareform(pdist(X[:, 1:], metric=self.metric))

# filter the euc_dist matrix using the time_dist
time_filter = math.pow(10, m)
dist = np.where(time_dist <= self.eps2, euc_dist, time_filter)

#%%
df['time'] = df['time'].dt.floor('min')
df.drop_duplicates('time', keep='first')
df['ts_minutes'] = df['time'].astype('int') / 60 / (10 ** 9)

X = np.column_stack((np.radians(df.loc[:, ['lon', 'lat']].values), df['ts_minutes'].astype('int').values))
x_id = df.loc[:, 'id'].astype('int').values



#%%
from st_dbscan import ST_DBSCAN
eps_km = 3
min_samp = 250
# execute ST_DBSCAN
st_dbscan = ST_DBSCAN(eps1=eps_km / 6371, eps2=240, min_samples=min_samp,
                      metric='euclidean', n_jobs=1)
st_dbscan.fit_frame_split(X, frame_size=10000)
results_dict = {'id': x_id, 'clust_id': st_dbscan.labels, 'lat': df['lat'].values, 'lon': df['lon'].values}
df_results = pd.DataFrame(results_dict)
#df_results = df_results[df_results['clust_id'] >= -1 ]

#%%
#%%
df['time'] = df['time'].dt.floor('min')
df.drop_duplicates('time', keep='first')
df['ts_minutes'] = df['time'].astype('int') / 60 / (10 ** 9)

X = np.column_stack((np.radians(df.loc[:, ['lon', 'lat']].values), df['ts_minutes'].astype('int').values))
x_id = df.loc[:, 'id'].astype('int').values

from st_optics import ST_OPTICS
eps_km = 3
min_samp = 250
# execute ST_DBSCAN
st_opt = ST_OPTICS(eps1=eps_km / 6371, eps2=600, min_samples=min_samp,
                      metric='euclidean', n_jobs=-1)
st_opt.fit(X)
results_dict = {'id': x_id, 'clust_id': st_opt.labels, 'lat': df['lat'].values, 'lon': df['lon'].values}
df_results = pd.DataFrame(results_dict)
#df_results = df_results[df_results['clust_id'] >= -1 ]

#%%
df_posits = gnact.clust.get_uid_posits(('636016432',), engine, end_time='2018-01-01')
# format data for dbscan
X = (np.radians(df.loc[:, ['lon', 'lat']].values))
x_id = df.loc[:, 'id'].values
#%%
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances[55000:])
plt.show()

#%%
from sklearn.metrics.pairwise import haversine_distances
clust_id_value='clust_id'
"""This function finds the center of a cluster from dbscan results,
and finds the average distance for each cluster point from its cluster center.
Returns a df."""
# make a new df from the df_results grouped by cluster id
# with an aggregation for min/max/count of times and the mean for lat and long
df_centers = (df_results.groupby(['clust_id'])
                        .agg({'time':[min, max, 'count'],
                              'lat':'mean',
                              'lon':'mean'})
                        .reset_index(drop=False))
df_centers.columns=['clust_id','time_min','time_max','total_clust_count','average_lat','average_lon']
# find the average distance from the centerpoint
# We'll calculate this by finding all of the distances between each point in
# df_results and the center of the cluster.  We'll then take the min and the mean.
haver_list = []
for i in df_centers[clust_id_value]:
    X = (np.radians(df_results[df_results[clust_id_value] == i]
                    .loc[:, ['lat', 'lon']].values))
    Y = (np.radians(df_centers[df_centers[clust_id_value] == i]
                    .loc[:, ['average_lat', 'average_lon']].values))
    haver_result = (haversine_distances(X, Y)) * 6371.0088  # km to radians
    haver_dict = {clust_id_value: i, 'average_dist_from_center': np.mean(haver_result)}
    haver_list.append(haver_dict)
# merge the haver results back to df_centers
haver_df = pd.DataFrame(haver_list)
df_centers = pd.merge(df_centers, haver_df, how='left', on=clust_id_value)

#%%
import folium

# plot the track
m = folium.Map(location=[df_posits.lat.median(), df_posits.lon.median()],
               zoom_start=4, tiles='OpenStreetMap')
points = list(zip(df_posits.lat, df_posits.lon))
folium.PolyLine(points).add_to(m)
# plot the clusters
df_centers = gnact.clust.calc_centers(df_results)
for row in df_centers.itertuples():
    folium.Marker(location=[row.average_lat, row.average_lon],
                  popup=[f"Cluster: {row.clust_id} \n"
                         f"Count: {row.total_clust_count}\n"
                         f"Average Dist from center {round(row.average_dist_from_center, 2)}\n"
                         f"Min Time: {row.time_min}\n"
                         f"Max Time: {row.time_max}"]
                  ).add_to(m)
print(f'Plotted {len(df_centers)} total clusters.')
m.save('map.html')
#$$
#%%
#
#
# # %%
# # format data for dbscan
# X = (np.radians(df.loc[:, ['lon', 'lat']].values))
# x_id = df.loc[:, 'id'].values
#
# # transform to numpy array
# data = np.column_stack((df['ts_minutes'].values, np.radians(df['lon']).values, np.radians(df['lat']).values))
#
#
# # %%
# def plot(data, labels):
#     colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
#               '#6a3d9a']
#
#     for i in range(-1, len(set(labels))):
#         if i == -1:
#             col = [0, 0, 0, 1]
#         else:
#             col = colors[i % len(colors)]
#
#             clust = data[np.where(labels == i)]
#             plt.scatter(clust[:, 0], clust[:, 1], c=[col], s=1)
#     plt.show()
#
#     return None
#
#
# plt.scatter(df['lon'], df['lat'])
# plt.show()
#
# # %%
# from st_optics import ST_OPTICS
#
# st_optics = ST_OPTICS(xi=0.9, eps2=480, min_samples=130,
#                       max_eps=5 / 6371.0088,
#                       n_jobs=-1)
# # %%
# tick = datetime.datetime.now()
# st_optics.fit_frame_split(data, frame_size=1000)
# lapse = datetime.datetime.now() - tick
# print(lapse)
# print(np.unique(st_optics.labels))
#
# plot(np.degrees(data[:, 1:]), st_optics.labels)
#
# # %%
# tick = datetime.datetime.now()
# from sklearn.cluster import DBSCAN
#
# # format data for dbscan
# X = (np.radians(df.loc[:, ['lon', 'lat']].values))
# x_id = df.loc[:, 'id'].values
# # execute sklearn's OPTICS
# # 5km in radians is max eps
# dbscan = DBSCAN(min_samples=200, eps=2 / 6371.0088, metric='euclidean',
#                 algorithm='kd_tree', n_jobs=1)
# dbscan.fit(X)
# results_dict = {'id': x_id, 'clust_id': dbscan.labels_}
#
# lapse = datetime.datetime.now() - tick
# print(lapse)
# print(np.unique(dbscan.labels_))
#
# plot(np.degrees(X), dbscan.labels_)
# # %%
#
#
# tick = datetime.datetime.now()
# from sklearn.cluster import OPTICS
#
# # format data for dbscan
# X = (np.radians(df.loc[:, ['lon', 'lat']].values))
# x_id = df.loc[:, 'id'].values
# # execute sklearn's OPTICS
# # 5km in radians is max eps
# optics = OPTICS(min_samples=250, max_eps=2 / 6371.0088, metric='euclidean', cluster_method='xi',
#                 algorithm='kd_tree', n_jobs=1)
# optics.fit(X)
# results_dict = {'id': x_id, 'clust_id': optics.labels_}
#
# lapse = datetime.datetime.now() - tick
# print(lapse)
# print(np.unique(optics.labels_))
# plot(np.degrees(X), optics.labels_)
#
# # %%
# import hdbscan
#
# tick = datetime.datetime.now()
# X = (np.radians(df.loc[:, ['lon', 'lat']].values))
# x_id = df.loc[:, 'id'].values
#
# clusterer = hdbscan.HDBSCAN(min_cluster_size=500, metric='euclidean',
#                             cluster_selection_epsilon=1 / 6371, cluster_selection_method='eom')
# clusterer.fit(X)
#
# lapse = datetime.datetime.now() - tick
# print(lapse)
# print(np.unique(clusterer.labels_))
# plot(np.degrees(X), clusterer.labels_)
#
# # %%
# # Authors: Shane Grigsby <refuge@rocktalus.com>
# #          Adrin Jalali <adrin.jalali@gmail.com>
# # License: BSD 3 clause
#
#
# from sklearn.cluster import OPTICS, cluster_optics_dbscan
# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Generate sample data
#
# np.random.seed(0)
# n_points_per_cluster = 250
#
# C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
# C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
# C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
# C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
# C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
# C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
# X = np.vstack((C1, C2, C3, C4, C5, C6))
#
# clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)
#
# # Run the fit
# clust.fit(X)
#
# labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
#                                    core_distances=clust.core_distances_,
#                                    ordering=clust.ordering_, eps=0.5)
# labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
#                                    core_distances=clust.core_distances_,
#                                    ordering=clust.ordering_, eps=2)
#
# space = np.arange(len(X))
# reachability = clust.reachability_[clust.ordering_]
# labels = clust.labels_[clust.ordering_]
#
# plt.figure(figsize=(10, 7))
# G = gridspec.GridSpec(2, 3)
# ax1 = plt.subplot(G[0, :])
# ax2 = plt.subplot(G[1, 0])
# ax3 = plt.subplot(G[1, 1])
# ax4 = plt.subplot(G[1, 2])
#
# # Reachability plot
# colors = ['g.', 'r.', 'b.', 'y.', 'c.']
# for klass, color in zip(range(0, 5), colors):
#     Xk = space[labels == klass]
#     Rk = reachability[labels == klass]
#     ax1.plot(Xk, Rk, color, alpha=0.3)
# ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
# ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
# ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
# ax1.set_ylabel('Reachability (epsilon distance)')
# ax1.set_title('Reachability Plot')
#
# # OPTICS
# colors = ['g.', 'r.', 'b.', 'y.', 'c.']
# for klass, color in zip(range(0, 5), colors):
#     Xk = X[clust.labels_ == klass]
#     ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
# ax2.set_title('Automatic Clustering\nOPTICS')
#
# # DBSCAN at 0.5
# colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
# for klass, color in zip(range(0, 6), colors):
#     Xk = X[labels_050 == klass]
#     ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
# ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
# ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')
#
# # DBSCAN at 2.
# colors = ['g.', 'm.', 'y.', 'c.']
# for klass, color in zip(range(0, 4), colors):
#     Xk = X[labels_200 == klass]
#     ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
# ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
# ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')
#
# plt.tight_layout()
# plt.show()
#
# ##%%
