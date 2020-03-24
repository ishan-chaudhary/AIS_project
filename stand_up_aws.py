#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:42:02 2020

@author: patrickmaus
"""
import aws_credentials as a_c
import psycopg2
from sqlalchemy import create_engine

#%%


aws_conn = psycopg2.connect(host=host,database=database, 
                        user=user,password=password)

aws_c = aws_conn.cursor()
if aws_c:
    print('Connection to AWS is good.'.format(database))
else: print('Connection failed.')
aws_c.close()
#%% local conn
database = 'ais_test'
local_conn = psycopg2.connect(host="localhost",database=database)
local_c = local_conn.cursor()
if local_c:
    print('Connection to {} is good.'.format(database))
else:
    print('Error connecting.')
local_c.close()




#%%
import aws_credentials as a_c
user = a_c.user
host = a_c.host
port = '5432'
database = 'aws_ais_clustering'
password = a_c.password

aws_engine = create_engine('postgresql://{}@{}:{}/{}'.format(user, host, port, database))

#%%
import aws_credentials as a_c
user = a_c.user
host = a_c.host
port = '5432'
database = 'aws_ais_clustering'
password = a_c.password

from sqlalchemy import create_engine

aws_engine = create_engine('postgresql://{}:{}@{}:{}/{}'.format(user, password, 
                                                                host, port, database))
aws_conn = psycopg2.connect(host=host,database=database, 
                        user=user,password=password)

aws_c = aws_conn.cursor()
if aws_c:
    print('Connection to AWS is good.'.format(database))
else: print('Connection failed.')
aws_c.close()
