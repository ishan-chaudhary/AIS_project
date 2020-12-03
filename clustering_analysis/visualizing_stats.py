import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
#%%
uid = '636016432'
df_results = pd.read_csv(f'results_{uid}.csv')

#%%
df_results = pd.read_csv('stats_clustering_results_all.csv')
df_results = df_results.reset_index(drop=False).rename({'Unnamed: 0':'method_full'},axis=1)
df_results['method_type'] = df_results.method_full.str.split('_').str.get(0)
df_results['params'] = df_results.method_full.str.split('_').str.slice(start=1).str.join('_')

#%%
g = sns.scatterplot(df_results.recall, df_results.precision, hue=df_results.method_type, size=df_results.proc_time)
plt.title(f'Precision and Recall for UID {uid} \n Colored by Method, Sized by Processing Time')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()


#%% stats
df_metrics = pd.read_csv('clustering_analysis/results_636016432.csv')

df_metrics = df_metrics.reset_index(drop=False).rename({'Unnamed: 0':'method_full'},axis=1)
df_metrics['method_type'] = df_metrics.method_full.str.split('_').str.get(0)
df_metrics['params'] = df_metrics.method_full.str.split('_').str.slice(start=1).str.join('_')

#%%
import matplotlib.colors as mcolors
tab_colors = mcolors.TABLEAU_COLORS
methods = df_metrics['method_type'].unique()
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
for i in len(methods):

    df_metrics['color'] = df_metrics[df_metrics['method_type']==methods[i]]


#%%
df_metrics = df_results
sns.set_style("darkgrid")
g = sns.scatterplot('recall', 'precision', hue='method_type', size='total_clusters',
                    data=df_metrics)
plt.title(f'Precision and Recall for All Data \n Colored by Method, Sized by Total Clusters')
plt.legend(bbox_to_anchor=(1.05, 1))

plt.show()
#%%

g = sns.scatterplot('recall', 'precision', hue='method_type', size='proc_time',
                    data=df_metrics)
sns.set_style("darkgrid")
plt.title(f'Precision and Recall for MSC ARUSHI \n Colored by Method, Sized by Processing Time')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.show()

#%%
import matplotlib.pyplot as plt
# although not used, Axes3D is required
from mpl_toolkits.mplot3d import Axes3D


def scatter_3d(df, x_col, y_col, z_col, highlight=None):
    sns.set(style="darkgrid")
    X = df[x_col]
    Y = df[y_col]
    Z = df[z_col]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # if you want to plot a particular set of hyperparameters, input its
    # index value and that parameter will be highlighted a different color.
    if highlight:
        ax.scatter(X[np.arange(len(X)) != highlight], Y[np.arange(len(Y)) != highlight],
                   Z[np.arange(len(Z)) != highlight], c='r', marker='o')
        ax.scatter(X[highlight], Y[highlight], Z[highlight], c='black', marker='D')
    else:
        ax.scatter(X, Y, Z)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    #plt.title('clustering_analysis Metrics Evaluation for {}'.format(value))
    ax.view_init(30, 160)
    plt.show()
#%%
scatter_3d(df_metrics[df_metrics['method_type']=='dbscan'],
           'total_clusters', 'average_nearest_site', 'f1')

#%%
scatter_3d(df_metrics[df_metrics['method_type']=='dbscan'],
           'total_clusters', 'average_nearest_site', 'f1')