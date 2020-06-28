# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

import random
from collections import defaultdict
from nltk import ngrams
import pandas as pd

# establish connections and get the edgelist from the database.
conn = gsta.connect_psycopg2(gsta_config.loc_cargo_full_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_full_params)

# df_edgelist is already sorted by time using the gsta.get_edgelist function
df_edgelist = gsta.get_edgelist(edge_table='cargo_edgelist', engine=loc_engine, loiter_time=2)
df_edgelist.to_csv('full_edgelist_for_nlp.csv')
#%% function definition
def get_mmsi_history(mmsi, df_edgelist, print=False):
    # make an df with all the edges for one mmsi
    df = df_edgelist[df_edgelist['mmsi']==mmsi]
    # get all of the previous ports from that mmsi as sample, except for the last port
    sample = df['Source'].iloc[:].str.replace(' ', '_').values
    # the last port is the target
    target = df['Target'].iloc[-1].replace(' ', '_')
    # concat all the samples into one string
    mmsi_hist = ''
    for s in sample:
        mmsi_hist = mmsi_hist + ' ' + s
    # add the target to the str
    mmsi_hist = mmsi_hist + ' ' + target
    if print==True:
        print(f'Previous {str(len(mmsi_hist.split())-1)} ports for {mmsi} are:', mmsi_hist.split()[:-1])
        print('Next port is:', target)
    return (mmsi_hist.strip())

def build_history(df_edgelist):
    # build a history that includes all port visits per mmsi as a dict with the mmsi as
    # the key and the strings of each port visited as the values
    # make sure to replace ' ' with '_' in the strings so multi-word ports are one str
    history = dict()
    # get all unique mmsis
    mmsis = df_edgelist['mmsi'].unique()
    for mmsi in mmsis:
        mmsi_edgelist = df_edgelist[df_edgelist['mmsi'] == mmsi]
        mmsi_str = ''
        # add all the sources from the source column
        for s in mmsi_edgelist['Source'].values:
            mmsi_str = mmsi_str + ' ' + (s.replace(' ', '_'))
        # after adding all the sources, we still need to add the last target.
        # adding all the sources will provide the history of all but the n-1 port
        mmsi_str = mmsi_str + ' ' + (mmsi_edgelist['Target'].iloc[-1].replace(' ', '_'))
        # only add this history to the dict if the len of the value (# of ports) is >= 2
        if len(mmsi_str.split()) >= 2:
            history[mmsi] = mmsi_str.strip()
    return history

def history_split(history, test_percent=.2):
    history_test = dict()
    history_train = dict()
    for k, v in history.items():
        if random.random() > test_percent:
            history_train[k] = v
        else:
            history_test[k] = v
    return history_train, history_test

def build_ngram_model(history, N):
    # first build a new dict from history that has at least N ports
    historyN = dict()
    for k, v in history.items():
        if len(v.split()) > N:
                historyN[k] = v.strip()
    # Create a placeholder for model that uses the default dict.
    #  the lambda:0 means any new key will have a value of 0
    model = defaultdict(lambda: defaultdict(lambda: 0))
    # build tuple of wN to pass to the model dict
    wordsN = ()
    for i in range(1, N + 1, 1):
        wordsN = wordsN + ('w' + str(i),)
    # Count frequency
    # in history, the key is the mmsi, the value is the string of ports visited
    for k, v in historyN.items():
    # we split each value and for each Ngram, we populate the model
    # each key is the N-1 ports, and the value is the last port.
    # in this way a trigram uses the first two ports to determine probability
    # the third port was vistied
        for wordsN in ngrams(v.split(), N):
            model[wordsN[:-1]][wordsN[-1]] += 1
    # transform the counts to probabilities and populate the model dict
    for key in model:
        total_count = float(sum(model[key].values()))
        for target in model[key]:
            model[key][target] /= total_count
    return model

def predict_ngram(mmsi_history, model, N, print=False):
    # check to see if the provided mmsi history has min N number of stops
    if len(mmsi_history.split()) < N :
        if print==True:
            print('MMSI History has fewer than N number of ports visited.')
            print('Cannot make a prediction')
        return None
    else:
        # add the last n ports (except for the last one) to a tuple to pass to the model
        words = ()
        for i in range(N, 1, -1):
            words = words + (mmsi_history.split()[-i],)
        # get the predicted port based on the model.  predicted is a dict
        predicted = dict(model[words])
        predicted = {k: v for k, v in sorted(predicted.items(), key=lambda item: item[1], reverse=True)}

        if print==True:
            print('Top ports (limited to 5) are:')
            # print results
            if len(predicted) >= 5:
                for p in sorted(predicted, key=predicted.get, reverse=True)[:5]:
                    print(p, predicted[p])
            else:
                for p in sorted(predicted, key=predicted.get, reverse=True):
                    print(p, predicted[p])

            # collect results for analysis
            if len(predicted) >= 5:
                for p in (sorted(predicted, key=predicted.get, reverse=True)[:5][0]):
                        if p == mmsi_history.split()[-1]:
                            print('TRUE!!!')

        return predicted

def evaluate_ngram(mmsi_history, predicted, top):
    if predicted == None or bool(predicted) == False:
        return None
    else:
        keys = list(predicted.keys())
        target = mmsi_history.split()[-1]
        if target in keys[:top]:
            return True
        else:
            return False



#%% build history and split to test/train with 20% test
history = build_history(df_edgelist)
history_train, history_test = history_split(history, test_percent=.2)
#%% accuracy eval

# choose the ranks that will count as a correct answer.
# for example, top=3 means that prediction is correct if the the target
# is in the top 3 predicted ports from the given ngram model.
top = 3

# build ngrams from 2grams (equivalent to markov chain) to 7grams
for N in range(2,8):
    # build model
    model = build_ngram_model(history_train, N)

    # iterate through mmsis from history_test and make a prediction for last port for each mmsi
    accuracy_dict = dict()
    for mmsi in list(history_test.keys()):
        # get the mmsi history
        mmsi_history = get_mmsi_history(mmsi, df_edgelist)
        # the the predicted dict of dicts using the mmmsi history and model
        predicted = predict_ngram(mmsi_history, model, N)
        # determine the result (True if prediction in top ranks, false if not,
        # None if the given Ngram could not make a prediction on the history).
        result = evaluate_ngram(mmsi_history, predicted, top=top)
        # add to tracking dictionary
        accuracy_dict[mmsi] = result

    # count up the trues, falses, and nones.
    df_accuracy = pd.DataFrame.from_dict(accuracy_dict, orient='index', columns=['result'])
    trues = df_accuracy['result'].sum()
    falses = len(df_accuracy[df_accuracy['result']==False])
    nones = df_accuracy['result'].isna().sum()
    # determine accuracy and precision
    accuracy = trues / (trues + falses + nones)
    precision = trues / (trues + falses)

    print(f'For N=={N} and top=={top}:, accuracy = {round(accuracy,5)} and precision = {round(precision,5)}')
    print(f'Trues={trues}, Falses={falses}, Nones={nones}')

#%% monte carlo simulation
results_dict = dict()
run_numb = 0
N = 4
history = build_history(df_edgelist)

while run_numb < 1000:
    history_train, history_test = history_split(history, test_percent=.2)
    model = build_ngram_model(history_train, N)

    # iterate through mmsis from history_test and make a prediction for last port for each mmsi
    accuracy_dict = dict()
    for mmsi in list(history_test.keys()):
        # get the mmsi history
        mmsi_history = get_mmsi_history(mmsi, df_edgelist)
        # the the predicted dict of dicts using the mmmsi history and model
        predicted = predict_ngram(mmsi_history, model, N)
        # determine the result (True if prediction in top ranks, false if not,
        # None if the given Ngram could not make a prediction on the history).
        result = evaluate_ngram(mmsi_history, predicted, top=top)
        # add to tracking dictionary
        accuracy_dict[mmsi] = result

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

#%%
# # build models
# n2_model = build_ngram_model(history_train, 2)
# n3_model = build_ngram_model(history_train, 3)
# n4_model = build_ngram_model(history_train, 4)
# n5_model = build_ngram_model(history_train, 5)


# #%%
# # randomly select an mmsi
# mmsi = random.choice(list(history_test.keys()))
# # get the mmsi history
# mmsi_history = get_mmsi_history(mmsi, df_edgelist)
#
# print('Bigram func')
# predict_ngram(mmsi_history, n2_model, 2)
# print()
#
# print('Trigram func')
# predict_ngram(mmsi_history, n3_model, 3)
# print()
#
# print('quadgram func')
# predict_ngram(mmsi_history, n4_model, 4)
# print()
#
# print('quintgram func')
# predict_ngram(mmsi_history, n5_model, 5)
#
# #%%
# # randomly select an mmsi
# mmsi = random.choice(list(history_test.keys()))
# # get the mmsi history
# mmsi_history = get_mmsi_history(mmsi, df_edgelist)
#
# predicted = predict_ngram(mmsi_history, n2_model, 2)
# result = evaluate_ngram(mmsi_history, predicted, 5)
# print(result)



