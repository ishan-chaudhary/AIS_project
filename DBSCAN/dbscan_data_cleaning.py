#!/usr/bin/env python
# coding: utf-8

import pandas as pd

ports_full = pd.read_csv('wpi.csv')
ports = ports_full[['index_no','port_name','latitude','longitude']]
ports = ports.rename(columns={'latitude':'lat','longitude':'lon'})


port_activity = pd.read_csv('port_activity_sample.csv')
port_activity.head()
port_activity.info()

ship_position = pd.read_csv('ship_position_sample.csv')
ship_position.head()
ship_position.info()
#%%
# merge together
df_full = pd.merge(port_activity, ship_position, how='left', on=['mmsi','time'])
df_full.to_csv('dbscan_data.csv')
print(df_full.head())

