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

df_edgelist = df_edgelist[['Source', 'Target', 'mmsi']]
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


# %% function definition
def get_uid_history(uid, df_edgelist, print=False):
    # make an df with all the edges for one uid
    df = df_edgelist[df_edgelist['uid'] == uid]
    # get all of the previous ports from that uid as sample, except for the last port
    sample = df['Source'].iloc[:].str.replace(' ', '_').values
    # the last port is the target
    target = df['Target'].iloc[-1].replace(' ', '_')
    # concat all the samples into one string
    uid_hist = ''
    for s in sample:
        uid_hist = uid_hist + ' ' + s
    # add the target to the str
    uid_hist = uid_hist + ' ' + target
    if print == True:
        print(f'Previous {str(len(uid_hist.split()) - 1)} ports for {uid} are:', uid_hist.split()[:-1])
        print('Next port is:', target)
    return (uid_hist.strip())


def build_history(df_edgelist):
    # build a history that includes all port visits per uid as a dict with the uid as
    # the key and the strings of each port visited as the values
    # make sure to replace ' ' with '_' in the strings so multi-word ports are one str
    history = dict()
    # get all unique uids
    uids = df_edgelist['uid'].unique()
    for uid in uids:
        uid_edgelist = df_edgelist[df_edgelist['uid'] == uid]
        uid_str = ''
        # add all the sources from the source column
        for s in uid_edgelist['Source'].values:
            uid_str = uid_str + ' ' + (s.replace(' ', '_'))
        # after adding all the sources, we still need to add the last target.
        # adding all the sources will provide the history of all but the n-1 port
        uid_str = uid_str + ' ' + (uid_edgelist['Target'].iloc[-1].replace(' ', '_'))
        # only add this history to the dict if the len of the value (# of ports) is >= 2
        if len(uid_str.split()) >= 2:
            history[uid] = uid_str.strip()
    return history

def build_history_multiproces(df_edgelist, df_max_len=100000, numb_workers=6):
    # need to split the edgelist to relatively equal pieces with complete uid histories
    numb_dfs = (len(df_edgelist) // df_max_len)
    uids = np.array(df_edgelist['uid'].unique())
    split_uids = np.array_split(uids, numb_dfs)
    list_df = []
    for split in split_uids:
        df_piece = df_edgelist[df_edgelist['uid'].isin(split)]
        list_df.append(df_piece)

    if __name__ == '__main__':
        with Pool(numb_workers) as p:
            history_pieces = p.map(build_history, list_df)

    # recombine pieces
    history = dict()
    for piece in history_pieces:
        for k, v in piece.items():
            history[k] = v


def history_split(history, test_percent=.2):
    history_test = dict()
    history_train = dict()
    for k, v in history.items():
        if random.random() > test_percent:
            history_train[k] = v
        else:
            history_test[k] = v
    return history_train, history_test



# %% build history and split to test/train with 20% test
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

history_orig = build_history_multiproces(df_edgelist)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

