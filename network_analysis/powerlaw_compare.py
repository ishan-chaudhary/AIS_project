import numpy as np
import pandas as pd
import networkx as nx

import powerlaw

# plotting
import matplotlib.pyplot as plt
from matplotlib import colors

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

# reload modules when making edits
from importlib import reload

reload(gsta)
#%%
# %%
conn = gsta.connect_psycopg2(gsta_config.colone_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.colone_cargo_params)
# %% get edgelist from database
df = gsta.get_edgelist(edge_table='cargo_edgelist_1km', engine=loc_engine, loiter_time=2)
g = nx.from_edgelist(df[['Source', 'Target']].values)
#%%


data = [x[1] for x in list(nx.degree(g))]

results = powerlaw.Fit(data)
print(results.power_law.alpha)
print(results.power_law.xmin)
R, p = results.distribution_compare('power_law', 'lognormal')
print(R, p)
powerlaw.plot_pdf(data, linear_bins=False)
plt.title("Power Law Plot")
plt.show()


plt.figure(figsize=(8,6))
plt.scatter(node_len.keys(), node_len.values(), c=[int(k[0]) for k in list(node_len.keys())],
            s=[int(k[5:7]) for k in list(node_len.keys()))
plt.show()