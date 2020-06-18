import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
# plotting
import matplotlib.pyplot as plt

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)


def get_edgelist(edge_table, engine, loiter_time=2):
    # select all edges from the database and join them with the port info from wpi
    # if the node is greater than 0 (not 0 which is open ocean or null)
    # and has a time diff less than 2 hours.  That should also eliminate ports a
    # ship transits through but does not actually stop at.
    # df_stops is a table of all ports where a ship was within 5km for more than 2 hours.
    # these are the "stops" we will use to build our edgelist.
    df_stops = pd.read_sql_query(f"""select edge.node, edge.arrival_time, 
                                 edge.depart_time, edge.time_diff,
                                 edge.destination, edge.position_count, edge.mmsi, 
                                 wpi.port_name
                                 from {edge_table} as edge, wpi as wpi
                                 where edge.node=wpi.index_no and
                                 edge.node > 0 and
                                 time_diff > '{str(loiter_time)} hours';""", engine)
    df_stops.sort_values(['mmsi', 'arrival_time'], inplace=True)

    # to build the edge list, we will take the pieces from stops for the current node and the next node
    df_list = pd.concat([df_stops.node, df_stops.port_name,
                         df_stops.node.shift(-1), df_stops.port_name.shift(-1),
                         df_stops.mmsi, df_stops.mmsi.shift(-1),
                         df_stops.depart_time, df_stops.arrival_time.shift(-1)], axis=1)
    # rename the columns
    df_list.columns = ['Source_id', 'Source', 'Target_id', 'Target',
                       'mmsi', 'target_mmsi', 'source_depart', 'target_arrival']
    # drop any row where the mmsi is not the same.
    # this will leave only the rows with at least 2 nodes with valid stops, making one valid edge.
    # The resulting df is the full edge list
    df_list = (df_list[df_list['mmsi'] == df_list['target_mmsi']]
               .drop('target_mmsi', axis=1))
    # this filters ou self-loops
    df_edgelist_full = df_list[df_list['Source_id'] != df_list['Target_id']]
    return df_edgelist_full


df_edgelist = get_edgelist(edge_table='cargo_edgelist', engine=loc_engine, loiter_time=2)

#%%
history_str = ''
for mmsi in df_edgelist['mmsi'].values:
    mmsi_edgelist = df_edgelist[df_edgelist['mmsi'] == mmsi]
    mmsi_str = ''
    for s in mmsi_edgelist['Source'].values:
        mmsi_str = mmsi_str + ' ' + (s.replace(' ', '_'))
    history_str = history_str + mmsi_str + '\n'

print(history_str)

#%%
import re
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter

class MarkovChain:
    def __init__(self):
        self.lookup_dict = defaultdict(list)

    def add_document(self, string):
        preprocessed_list = self._preprocess(string)
        pairs = self.__generate_tuple_keys(preprocessed_list)
        for pair in pairs:
            self.lookup_dict[pair[0]].append(pair[1])
        pairs2 = self.__generate_2tuple_keys(preprocessed_list)
        for pair in pairs2:
            self.lookup_dict[tuple([pair[0], pair[1]])].append(pair[2])
        pairs3 = self.__generate_3tuple_keys(preprocessed_list)
        for pair in pairs3:
            self.lookup_dict[tuple([pair[0], pair[1], pair[2]])].append(pair[3])

    def _preprocess(self, string):
        cleaned = re.sub(r'\W+', ' ', string).lower()
        tokenized = word_tokenize(cleaned)
        return tokenized

    def __generate_tuple_keys(self, data):
        if len(data) < 1:
            return

        for i in range(len(data) - 1):
            yield [data[i], data[i + 1]]

    def __generate_2tuple_keys(self, data):
        if len(data) < 2:
            return

        for i in range(len(data) - 2):
            yield [data[i], data[i + 1], data[i + 2]]

    def __generate_3tuple_keys(self, data):
        if len(data) < 3:
            return

        for i in range(len(data) - 3):
            yield [data[i], data[i + 1], data[i + 2], data[i + 3]]

    def oneword(self, string):
        return Counter(self.lookup_dict[string]).most_common()[:3]

    def twowords(self, string):
        suggest = Counter(self.lookup_dict[tuple(string)]).most_common()[:3]
        if len(suggest) == 0:
            return self.oneword(string[-1])
        return suggest

    def threewords(self, string):
        suggest = Counter(self.lookup_dict[tuple(string)]).most_common()[:3]
        if len(suggest) == 0:
            return self.twowords(string[-2:])
        return suggest

    def morewords(self, string):
        return self.threewords(string[-3:])

    def generate_text(self, string):
        if len(self.lookup_dict) > 0:
            tokens = string.split(" ")
            if len(tokens) == 1:
                print("Next word suggestions:", self.oneword(string))
            elif len(tokens) == 2:
                print("Next word suggestions:", self.twowords(string.split(" ")))
            elif len(tokens) == 3:
                print("Next word suggestions:", self.threewords(string.split(" ")))
            elif len(tokens) > 3:
                print("Next word suggestions:", self.morewords(string.split(" ")))
        return

#%%
mmsi = df_edgelist['mmsi'].sample().values[0]
df = df_edgelist[df_edgelist['mmsi']==mmsi]
sample = df['Source'].iloc[:-1].str.replace(' ', '_').values
target = df['Source'].iloc[-1].replace(' ', '_')
sample_str = ''
for s in sample:
    sample_str = s + ' ' + sample_str
print('Previous ports are:', sample_str)
print('Next port is:', target)

my_markov = MarkovChain()
my_markov.add_document(history_str)
my_markov.generate_text(sample_str.strip().lower())

