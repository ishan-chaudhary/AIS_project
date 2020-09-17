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
# %%
# conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.colone_cargo_params)

# %%
df = pd.read_sql_query("SELECT id, time, lat, lon FROM ais_cargo.public.uid_positions "
                       "WHERE uid = '636016432'"
                       "ORDER BY time", loc_engine)
df.to_csv('sample_ship_posits.csv')
#%%

df2 = pd.read_sql_query("SELECT id, time, lat, lon FROM ais_cargo.public.uid_positions_status "
                       "WHERE uid = '636016432'"
                       "ORDER BY time", loc_engine)


# %%
import math
from sklearn.metrics.pairwise import haversine_distances



def calc_bearing(origin_lat, origin_lon, dest_lat, dest_lon):
    delta_lon = dest_lon - origin_lon
    delta_lat = dest_lat - origin_lat
    bearing = math.atan2(delta_lon, delta_lat) / math.pi * 180
    if bearing < 0:
        bearing_final = 360 + bearing
    else:
        bearing_final = bearing
    return bearing_final

def calc_distance(origin_lat, origin_lon, dest_lat, dest_lon):
    origin_rads = np.radians([origin_lat, origin_lon])
    dest_rads = np.radians([dest_lat, dest_lon])
    result = haversine_distances([origin_rads, dest_rads])
    result_km = result * 6371000 / 1000
    return result_km[0,1]



df['cog'] =df.apply(calc_bearing(df.previous_lat, df.previous_lon, df.lat, df.lon))
print(calc_distance(df.previous_lat[1], df.previous_lon[1], df.lat[1], df.lon[1]))
# %%
# if we want to round by minute
df['time_rounded'] = df['time'].apply(lambda x: x.floor('min'))

# transferring datetime to int gives nanonseconds since time zero.
# this transformation will give you time since zero IN MINUTES
df['ts_minutes'] = df['time'].astype('int') / 60 / (10 ** 9)

df['time_diff'] = df['time'] - df['time'].shift(1)
df['previous_lat'] = df['lat'].shift(1)
df['previous_lon'] = df['lon'].shift(1)

# %%
# format data for dbscan
X = (np.radians(df.loc[:, ['lon', 'lat']].values))
x_id = df.loc[:, 'id'].values

# transform to numpy array
data = np.column_stack((df['ts_minutes'].values, np.radians(df['lon']).values, np.radians(df['lat']).values))


# %%
def plot(data, labels):
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
              '#6a3d9a']

    for i in range(-1, len(set(labels))):
        if i == -1:
            col = [0, 0, 0, 1]
        else:
            col = colors[i % len(colors)]

            clust = data[np.where(labels == i)]
            plt.scatter(clust[:, 0], clust[:, 1], c=[col], s=1)
    plt.show()

    return None


plt.scatter(df['lon'], df['lat'])
plt.show()

# %%
from st_optics import ST_OPTICS

st_optics = ST_OPTICS(xi=0.9, eps2=480, min_samples=130,
                      max_eps=5 / 6371.0088,
                      n_jobs=-1)
# %%
tick = datetime.datetime.now()
st_optics.fit_frame_split(data, frame_size=1000)
lapse = datetime.datetime.now() - tick
print(lapse)
print(np.unique(st_optics.labels))

plot(np.degrees(data[:, 1:]), st_optics.labels)

# %%
tick = datetime.datetime.now()
from sklearn.cluster import DBSCAN

# format data for dbscan
X = (np.radians(df.loc[:, ['lon', 'lat']].values))
x_id = df.loc[:, 'id'].values
# execute sklearn's OPTICS
# 5km in radians is max eps
dbscan = DBSCAN(min_samples=200, eps=2 / 6371.0088, metric='euclidean',
                algorithm='kd_tree', n_jobs=1)
dbscan.fit(X)
results_dict = {'id': x_id, 'clust_id': dbscan.labels_}

lapse = datetime.datetime.now() - tick
print(lapse)
print(np.unique(dbscan.labels_))

plot(np.degrees(X), dbscan.labels_)
# %%


tick = datetime.datetime.now()
from sklearn.cluster import OPTICS

# format data for dbscan
X = (np.radians(df.loc[:, ['lon', 'lat']].values))
x_id = df.loc[:, 'id'].values
# execute sklearn's OPTICS
# 5km in radians is max eps
optics = OPTICS(min_samples=250, max_eps=2 / 6371.0088, metric='euclidean', cluster_method='xi',
                algorithm='kd_tree', n_jobs=1)
optics.fit(X)
results_dict = {'id': x_id, 'clust_id': optics.labels_}

lapse = datetime.datetime.now() - tick
print(lapse)
print(np.unique(optics.labels_))
plot(np.degrees(X), optics.labels_)

# %%
import hdbscan

tick = datetime.datetime.now()
X = (np.radians(df.loc[:, ['lon', 'lat']].values))
x_id = df.loc[:, 'id'].values

clusterer = hdbscan.HDBSCAN(min_cluster_size=500, metric='euclidean',
                            cluster_selection_epsilon=1 / 6371, cluster_selection_method='eom')
clusterer.fit(X)

lapse = datetime.datetime.now() - tick
print(lapse)
print(np.unique(clusterer.labels_))
plot(np.degrees(X), clusterer.labels_)

# %%
# Authors: Shane Grigsby <refuge@rocktalus.com>
#          Adrin Jalali <adrin.jalali@gmail.com>
# License: BSD 3 clause


from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data

np.random.seed(0)
n_points_per_cluster = 250

C1 = [-5, -2] + .8 * np.random.randn(n_points_per_cluster, 2)
C2 = [4, -1] + .1 * np.random.randn(n_points_per_cluster, 2)
C3 = [1, -2] + .2 * np.random.randn(n_points_per_cluster, 2)
C4 = [-2, 3] + .3 * np.random.randn(n_points_per_cluster, 2)
C5 = [3, -2] + 1.6 * np.random.randn(n_points_per_cluster, 2)
C6 = [5, 6] + 2 * np.random.randn(n_points_per_cluster, 2)
X = np.vstack((C1, C2, C3, C4, C5, C6))

clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)

# Run the fit
clust.fit(X)

labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.5)
labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)

space = np.arange(len(X))
reachability = clust.reachability_[clust.ordering_]
labels = clust.labels_[clust.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[clust.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'k+', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = X[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = X[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

plt.tight_layout()
plt.show()

##%%
