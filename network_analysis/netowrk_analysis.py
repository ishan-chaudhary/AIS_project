#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:15:33 2020

@author: patrickmaus
"""

import psycopg2
import pandas as pd
import datetime
import networkx as nx

G = nx.MultiDiGraph()
G.add_edges_from(edge_list)

#%%
nx.draw(G)
nx.draw_networkx(G)