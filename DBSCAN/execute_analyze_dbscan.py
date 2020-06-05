#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:11:39 2020

@author: patrickmaus
"""

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config
import pandas as pd

#pandas and postgres time interval types dont mesh well so every read_table
#call returns a warning
import warnings 
warnings.filterwarnings('ignore')

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_params)


#%% first round params and make schema
#first_round_params = [[.5, 50], [.5, 100], [.25,25], [.25, 50], [.25,100], [1,100]]

# make the new schema for todays date and the method.
#schema_name = gsta.create_schema('sklearn_mmsi', conn, drop_schema=False, with_date=False)  

#%% first round params for samples
epsilons_km = [.25, .5, 1, 2, 3, 5, 7]
samples = [2, 5, 10, 25, 50, 100, 250, 500]

first_round_params = []
for eps_km in epsilons_km:
    for min_samples in samples: 
        first_round_params.append([eps_km, min_samples])

# make the new schema the method.
first_round_schema_name = gsta.create_schema('sklearn_sample_first_rnd', conn, drop_schema=True, with_date=False)  

                   
#%% first round dbscan execute
# run the first round of dbscan against the source data, storing in the first_round schema
gsta.execute_dbscan(source_table='ship_position_sample', 
                    to_schema_name=first_round_schema_name,
                    from_schema_name='public',
                    method='sklearn_mmsi',
                    eps_samples_params=first_round_params,
                    conn=conn, engine=loc_engine)
# analyze the first round results, writing to the same schema
gsta.analyze_dbscan(method_used='sklearn_mmsi', 
                    schema_name=first_round_schema_name, 
                    ports_labeled='ports_5k_sample_positions', 
                    eps_samples_params=first_round_params,
                    id_value='mmsi', clust_id_value='clust_id',
                    noise_filter=10, conn=conn, engine=loc_engine)

#%% second round params
epsilons_km = [.5, 1, 2, 3, 5]
samples = [2, 3, 4, 5, 7, 10]

second_round_params = []
for eps_km in epsilons_km:
    for min_samples in samples: 
        second_round_params.append([eps_km, min_samples])

#%% Execute second round of DBSCAN
second_round_df = pd.DataFrame()
for p in first_round_params:
    
    #break out the eps and min samples from the list
    eps_km, min_samples = p
    # create the second round schema
    second_round_schema = gsta.create_schema('sklearn_sample_second_rnd' + '_' 
                                             + str(eps_km).replace('.','_')
                                             + '_' + str(min_samples),
                                             conn, drop_schema=True, with_date=False)  
    # define the source table, which will change for each iteration of params
    source_table = ('summary_' + str(eps_km).replace('.','_') +
                    '_' + str(min_samples))
    # execute dbscan against the first round results
    gsta.execute_dbscan(source_table=source_table, 
                        from_schema_name=first_round_schema_name,
                        to_schema_name=second_round_schema,
                        method='sklearn_rollup',
                        eps_samples_params=second_round_params,
                        conn=conn, engine=loc_engine)
    # analyze the second round results
    df = gsta.analyze_dbscan(method_used='sklearn_rollup',
                     schema_name=second_round_schema,
                     ports_labeled='ports_5k_sample_positions', 
                     eps_samples_params=second_round_params,
                     id_value='id', clust_id_value='super_clust_id',
                     noise_filter=1000, conn=conn, engine=loc_engine)
    # some of the params are none because they had zero clusters.  
    # try converting to string.  if fail, pass.
    try:
        df['first_round_params'] = (str(p[0]) + '_' + str(p[1]))
    except: pass
    second_round_df = second_round_df.append(df)
    
