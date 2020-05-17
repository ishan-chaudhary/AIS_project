#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:11:39 2020

@author: patrickmaus
"""

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

import warnings
warnings.filterwarnings('ignore')

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)


#%% first round params and make schema
first_round_params = [[.5, 50], [.5, 100], [.25,25], [.25, 50], [.25,100], [1,100]]

# make the new schema for todays date and the method.
schema_name = gsta.create_schema('sklearn_mmsi', conn, drop_schema=False, with_date=False)  

                   
#%% first round dbscan execute
gsta.execute_dbscan(source_table='cargo_ship_position', 
                    to_schema_name='sklearn_mmsi_2020_05_15',
                    from_schema_name='public',
                    method='sklearn_mmsi',
                    eps_samples_params=first_round_params,
                    conn=conn, engine=loc_engine)

#%% first round analysis
gsta.analyze_dbscan(method_used='sklearn_mmsi', 
                    schema_name='sklearn_mmsi_2020_05_15', 
                    ports_labeled='ports_5k_positions', 
                    eps_samples_params=[[.5, 50], [.5, 100]],
                    id_value='mmsi', clust_id_value='clust_id',
                    conn=conn, engine=loc_engine)

#%% second round params
epsilons_km = [.5, 1, 2, 3, 5]
samples = [2, 3, 4, 5, 7, 10]

second_round_params = []
for eps_km in epsilons_km:
    for min_samples in samples: 
        second_round_params.append([eps_km, min_samples])

#%% Execute second round of DBSCAN
for p in [[.5, 50], [.5, 100]]:
    
    eps_km, min_samples = p
    second_round_schema = gsta.create_schema('sklearn_second_rnd_' + str(eps_km).replace('.','_')
                                       + '_' + str(min_samples),
                                       conn, drop_schema=True, with_date=False)  
    source_table = ('summary__' + str(eps_km).replace('.','_') +
                    '_' + str(min_samples))
    gsta.execute_dbscan(source_table=source_table, 
                        from_schema_name='sklearn_mmsi_2020_05_15',
                        to_schema_name=second_round_schema,
                        method='sklearn_rollup',
                        eps_samples_params=second_round_params,
                        conn=conn, engine=loc_engine)

#%%    
second_round_df = pd.DataFrame()
for p in [[.5, 50], [.5, 100]]:
    
    eps_km, min_samples = p
    second_round_schema = gsta.create_schema('sklearn_second_rnd_' + str(eps_km).replace('.','_')
                                       + '_' + str(min_samples),
                                       conn, drop_schema=False, with_date=False)  
    source_table = ('summary__' + str(eps_km).replace('.','_') +
                    '_' + str(min_samples))

    df = gsta.analyze_dbscan(method_used='sklearn_rollup',
                     schema_name=second_round_schema,
                     ports_labeled='ports_5k_positions', 
                     eps_samples_params=second_round_params,
                     id_value='id', clust_id_value='super_clust_id',
                     conn=conn, engine=loc_engine)
    # some of the params are none because they had zero clusters.  
    # try converting to string.  if fail, pass.
    try:
        df['first_round_params'] = (df['eps_km'].astype('str') + '_'
                          + df['min_samples'].astype('str'))
    except: pass
    second_round_df = second_round_df.append(df)
