
from collections import defaultdict
from nltk import ngrams
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import random

import gsta  # geospatial analysis tools

from multiprocessing import Pool

# %%
sep = pd.read_csv('air_data/sep_2019.csv')
oct = pd.read_csv('air_data/oct_2019.csv')
nov = pd.read_csv('air_data/nov_2019.csv')
dec = pd.read_csv('air_data/dec_2019.csv')
jan = pd.read_csv('air_data/jan_2020.csv')
feb = pd.read_csv('air_data/feb_2020.csv')

# put together the month pieces
df_flights = pd.concat([sep, oct, nov, dec, jan, feb])
del (sep, oct, nov, dec, jan, feb)
# %%
df_flights['tail_flight_combo'] = df_flights['TAIL_NUM'] + '_' + df_flights['OP_CARRIER_FL_NUM'].astype('str')
df_flights['carrier_flight_combo'] = df_flights['OP_UNIQUE_CARRIER'] + '_' + df_flights['OP_CARRIER_FL_NUM'].astype('str')
# need to sort so rows are sequential
df_flights = df_flights.sort_values(['carrier_flight_combo', 'FL_DATE', 'DEP_TIME'])
# %%
df_edgelist = pd.concat([df_flights['ORIGIN'], df_flights['DEST'], df_flights['tail_flight_combo']], axis=1)
# Source and Target should be capitilzed so the network can be read into Gephi.
df_edgelist.columns = ['Source', 'Target', 'uid']
df_edgelist.dropna(inplace=True)

print(f'The total number of rows is {len(df_edgelist)}.')
print(f"The total of unique UIDs is {len(df_edgelist['uid'].unique())}.")

df_sample_uid = df_edgelist[df_edgelist['uid'] == 'N13903']

# %%
import importlib
importlib.reload(gsta)

# %% build history and split to test/train with 20% test
# noinspection DuplicatedCode
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

# need to split the edgelist to relatively equal pieces with complete uid histories
df_max_len = 100000
numb_workers = 6
numb_dfs = (len(df_edgelist) // df_max_len)
uids = np.array(df_edgelist['uid'].unique())
split_uids = np.array_split(uids, numb_dfs)
list_df = []
for split in split_uids:
    df_piece = df_edgelist[df_edgelist['uid'].isin(split)]
    list_df.append(df_piece)

if __name__ == '__main__':
    with Pool(numb_workers) as p:
        history_pieces = p.map(gsta.build_history, list_df)

# recombine pieces
history = dict()
for piece in history_pieces:
    for k, v in piece.items():
        history[k] = v

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)


#%%
length_dict = {key: len(value.split()) for key, value in history.items()}
fig, ax = plt.figure()
plt.hist(length_dict)
plt.show()

# %% accuracy eval
# choose the ranks that will count as a correct answer.
# for example, top=3 means that prediction is correct if the the target
# is in the top 3 predicted ports from the given ngram model.
top = 3

# build an 80/20 train/test split from history
history_train, history_test = gsta.history_split(history, test_percent=.2)

# build ngrams from 2grams (equivalent to markov chain) to 7grams
for N in range(2, 10):
    # build model
    model = gsta.build_ngram_model(history_train, N)

    # iterate through uids from history_test and make a prediction for last port for each uid
    accuracy_dict = dict()
    for uid in list(history_test.keys()):
        # get the uid history
        uid_history = gsta.get_uid_history(uid, df_edgelist)
        # the the predicted dict of dicts using the muid history and model
        predicted = gsta.predict_ngram(uid_history, model, N)
        # determine the result (True if prediction in top ranks, false if not,
        # None if the given Ngram could not make a prediction on the history).
        result = gsta.evaluate_ngram(uid_history, predicted, top=top)
        # add to tracking dictionary
        accuracy_dict[uid] = result

    # count up the trues, falses, and nones.
    df_accuracy = pd.DataFrame.from_dict(accuracy_dict, orient='index', columns=['result'])
    trues = df_accuracy['result'].sum()
    falses = len(df_accuracy[df_accuracy['result'] == False])
    nones = df_accuracy['result'].isna().sum()
    # determine accuracy and precision
    accuracy = trues / (trues + falses + nones)
    precision = trues / (trues + falses)

    print(f'For N=={N} and top=={top}:, accuracy = {round(accuracy, 5)} and precision = {round(precision, 5)}')
    print(f'Trues={trues}, Falses={falses}, Nones={nones}')

# %% monte carlo simulation
results_dict = dict()
run_numb = 0  # should be set to zero
max_runs = 1  # note that anything above 10 can take minutes to complete
N = 3  # N gram to model and simulate accuracy
top = 3  # rank required to be a correct answer

# conduct multiple runs to simulate a large number of train/test splits
while run_numb < max_runs:
    history_train, history_test = gsta.history_split(history, test_percent=.2)
    model = gsta.build_ngram_model(history_train, N)

    # iterate through uids from history_test and make a prediction for last port for each uid
    accuracy_dict = dict()
    for uid in list(history_test.keys()):
        # get the uid history
        uid_history = gsta.get_uid_history(uid, df_edgelist)
        # the the predicted dict of dicts using the muid history and model
        predicted = gsta.predict_ngram(uid_history, model, N)
        # determine the result (True if prediction in top ranks, false if not,
        # None if the given Ngram could not make a prediction on the history).
        result = gsta.evaluate_ngram(uid_history, predicted, top=top)
        # add to tracking dictionary
        accuracy_dict[uid] = result

    # count up the trues, falses, and nones.
    df_accuracy = pd.DataFrame.from_dict(accuracy_dict, orient='index', columns=['result'])
    trues = df_accuracy['result'].sum()
    falses = len(df_accuracy[df_accuracy['result'] == False])
    nones = df_accuracy['result'].isna().sum()
    # determine accuracy and precision
    accuracy = trues / (trues + falses + nones)
    precision = trues / (trues + falses)

    results_dict[run_numb] = [trues, falses, nones, accuracy, precision]
    run_numb += 1

df_results = pd.DataFrame.from_dict(results_dict, orient='index',
                                    columns=['trues', 'falses', 'nones', 'accuracy', 'precision'])
print(f'Over {run_numb} runs with N=={N}: '
      f'Average accuracy={round(df_results.accuracy.mean(), 5)} and '
      f'Average precision={round(df_results.precision.mean(), 5)}')


# %% predicting time to arrive
def predict_time(previous_port, next_port, df):
    df_set = df[(df['Source'] == previous_port) & (df['Target'] == next_port)]
    df_set['time_diff'] = df_set.loc[:, 'target_arrival'] - df_set.loc[:, 'source_depart']

    df_set['time_diff'].astype('timedelta64[h]').plot.hist()
    median = df_set['time_diff'].astype('timedelta64[h]').median()
    plt.title(f'Histogram of Travel Time in Hours from {previous_port.title()} to {next_port.title()}')
    plt.figtext(.05, .02, f'Dashed line is median value, {median} hours.')
    plt.xlabel('Hours')
    plt.axvline(median, color='k', linestyle='dashed', linewidth=1)
    plt.show()

    print('Total observations:', len(df_set['time_diff']))
    print('Median:', df_set['time_diff'].astype('timedelta64[h]').median())
    print('Mean:', df_set['time_diff'].astype('timedelta64[h]').mean())
    print('Minimum:', df_set['time_diff'].astype('timedelta64[h]').min())
    print('Maximum:', df_set['time_diff'].astype('timedelta64[h]').max())


previous_port = 'GLOUCESTER'
next_port = 'NEWARK'

predict_time(previous_port, next_port, df_edgelist)

# %%
# # %% experimentation with predicting for every ngram withing uid history, not just last port
# port_list = list()
# for k, v in history.items():
#     for words in ngrams(v.split(), 3):
#         if words not in port_list:
#             port_list.append(words)
#         else:
#             continue
#
# # %%
# for p in port_list[:5]:
#     print('Previous ports are:', p[0], p[1])
#     print('Target port is:', p[2])
#     for k, v in (model[p[0], p[1]]).items():
#         print(k)
#         print(v)

# %% early dev work
# # build models
# n2_model = build_ngram_model(history_train, 2)
# n3_model = build_ngram_model(history_train, 3)
# n4_model = build_ngram_model(history_train, 4)
# n5_model = build_ngram_model(history_train, 5)


# #%%
# # randomly select an uid
# uid = random.choice(list(history_test.keys()))
# # get the uid history
# uid_history = get_uid_history(uid, df_edgelist)
#
# print('Bigram func')
# predict_ngram(uid_history, n2_model, 2)
# print()
#
# print('Trigram func')
# predict_ngram(uid_history, n3_model, 3)
# print()
#
# print('quadgram func')
# predict_ngram(uid_history, n4_model, 4)
# print()
#
# print('quintgram func')
# predict_ngram(uid_history, n5_model, 5)
#
# #%%
# # randomly select an uid
# uid = random.choice(list(history_test.keys()))
# # get the uid history
# uid_history = get_uid_history(uid, df_edgelist)
#
# predicted = predict_ngram(uid_history, n2_model, 2)
# result = evaluate_ngram(uid_history, predicted, 5)
# print(result)
