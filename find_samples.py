#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:06:44 2019

@author: patrickmaus
"""

import psycopg2
import random
import pandas as pd

#%% Establish connection and test
conn = psycopg2.connect(host="localhost",database="ais_data")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()
#%% Select random samples
c = conn.cursor()
c.execute('select distinct mmsi from ship_position')
mmsi_list = c.fetchall()
random.seed(1)
mmsi_sample = random.sample(mmsi_list, 10)
#%% Select all data for each mmsi and make a new table with sample data
for m in mmsi_sample:
    
    c = conn.cursor()
    c.execute("""select count(*) from ship_position 
              where mmsi = '{}'""".format(m[0]))
    m_count = c.fetchone()[0]
    print('Getting {} records for MMSI {}...'.format(m_count, m[0]))
    
    c.execute("""insert into ship_position_sample
              select * from ship_position 
              where mmsi = '{}'""".format(m[0]))
    conn.commit()
    c.close()
    print('{} added'.format(m[0]))

