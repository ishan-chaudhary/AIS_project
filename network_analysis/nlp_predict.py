# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)

df_edgelist = gsta.get_edgelist(edge_table='cargo_edgelist', engine=loc_engine, loiter_time=2)

#%%
history = dict()
mmsis = df_edgelist['mmsi'].unique()
for mmsi in mmsis:
    mmsi_edgelist = df_edgelist[df_edgelist['mmsi'] == mmsi]
    mmsi_str = ''
    # add all the sources
    for s in mmsi_edgelist['Source'].values:
        mmsi_str = mmsi_str + ' ' + (s.replace(' ', '_'))
    # after adding all the sources, we still need to add the last target.
    # adding all the sources will provide the history of all but the n-1 port
    mmsi_str = mmsi_str + ' ' + (mmsi_edgelist['Target'].iloc[-1].replace(' ', '_'))
    if len(mmsi_str.split()) > 3:
        history[mmsi] = mmsi_str.strip()

#%%
import random
from nltk import ngrams
from collections import defaultdict

# Create a placeholder for model
model = defaultdict(lambda: defaultdict(lambda: 0))

# Count frequency of cooccurance
for k, v in history.items():
    for w1, w2, w3 in ngrams(v.split(), 3):
        model[(w1, w2)][w3] += 1

# transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

#%% testing
# randomly select an mmsi
mmsi = random.choice(list(history.keys()))
# make an edgelist
df = df_edgelist[df_edgelist['mmsi']==mmsi]
# get all of the previous ports from that mmsi as sample, except for the last
sample = df['Source'].iloc[:].str.replace(' ', '_').values
# the last port is the target
target = df['Target'].iloc[-1].replace(' ', '_')
# concat all the samples into one string
sample_str = ''
for s in sample:
    sample_str = sample_str + ' ' + s
print('Previous ports are:', sample_str)
print('Next port is:', target)

# get the predicted port based on the model.  predicted is a dict
predicted = dict(model[sample_str.split()[-2], sample_str.split()[-1]])
print(predicted)

if target in predicted.keys():
    print('True!')
