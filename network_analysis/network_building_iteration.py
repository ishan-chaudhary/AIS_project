import pandas as pd
import datetime

# Geo-Spatial Temporal Analysis package
import gsta
import gsta_config

conn = gsta.connect_psycopg2(gsta_config.loc_cargo_full_params)
loc_engine = gsta.connect_engine(gsta_config.loc_cargo_full_params)

#%% variable delcaration
port_activity_table = 'nearest_port'
edge_table = 'cargo_edgelist'
dist = 5

#%% port check function
def port_check(row, dist=dist):
    if row['nearest_port_dist_km'] <=dist:
        val = row['nearest_port_id']
    else:
        val = 0
    return val

#%% Create the edge table
c = conn.cursor()
#c.execute("""DROP TABLE IF EXISTS {};""".format(edge_table))
conn.commit()
c.execute("""CREATE TABLE IF NOT EXISTS {}  (
        node                int,
        arrival_time        timestamp,
        depart_time         timestamp,
        time_diff           interval,
        destination         int,
        position_count      bigint,
        mmsi                text
        )         
""".format(edge_table))
conn.commit()
c.close()

#%% set indices for tables.
c = conn.cursor()
conn.commit()
c.execute(f"""CREATE INDEX port_mmsi_idx on {port_activity_table} (mmsi);""")
conn.commit()
c.close()

c = conn.cursor()
conn.commit()
c.execute(f"""CREATE INDEX edgelist_mmsi_idx on {edge_table} (mmsi);""")
conn.commit()
c.close()


#%% get mmsi lists

# This list is all of the mmsis in the table of interest.  It is the
# total number of mmsis we will be iterating over.
c = conn.cursor()
c.execute("""SELECT DISTINCT(mmsi) FROM {} 
          WHERE nearest_port_dist_km < {};"""
          .format(port_activity_table, dist))
mmsi_list_potential = c.fetchall()
c.close()

# if we have to stop the process, we can use the mmsis that are already completed
# to build a new list of mmsis left to complete.  this will allow us to resume
# processing without repeating any mmsis.
c = conn.cursor()
c.execute("""SELECT DISTINCT(mmsi) FROM {};"""
          .format(edge_table))
mmsi_list_completed = c.fetchall()
c.close()

# find the mmsis that are not in the edge table yet
diff = lambda l1,l2: [x for x in l1 if x not in l2]
mmsi_list = diff(mmsi_list_potential, mmsi_list_completed)
#%% iterate through the mmsi list and build the network edges
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

# run count = total mmsis already processed plus current batch
run_count = len(mmsi_list_completed)

for i in range(len(mmsi_list)): #iterate through all the mmsi #'s gathered
    # get the mmsi from the tuple
    mmsi = mmsi_list[i]
    
    print('Working on MMSI:', mmsi[0])
        
    # not efficent, but the easiest way to parse.  Largest number of positions
    # for all data is only about 40,000, so pandas can handle it
    df = pd.read_sql("""select time, nearest_port_id, nearest_port_dist_km 
                     from {0} 
                     where mmsi = '{1}'
                     order by time""".format(port_activity_table, mmsi[0]), loc_engine)

    # port_check takes the dist to nearest port and if its less than dist, populates
    # port_id with the nearest port id.  If the nearest port is greater than dist,
    # port_id = 0.  0 will be used for activity "not in port"
    df['node'] = df.apply(port_check, axis=1)
    df.drop(['nearest_port_id', 'nearest_port_dist_km'], axis=1, inplace=True)
    # use shift to get the next node and previous node
    df['next_node'] = df['node'].shift(-1)
    df['prev_node'] = df['node'].shift(1)
    # reduce the dataframe down to only the positions where the previous node is
    # different from the next node.  These are the transitions between nodes 
    df_reduced = (df[df['next_node'] != df['prev_node']]
                  .reset_index())
    # make a df of all the starts and all the ends.  When the node is the same as
    # the next node (but next node is different than previous node), its the start
    # of actiivity at a node.  Similarly, when the node is the same as the previous
    # node (but the next node is different than previous node), its the end of activity.
    df_starts = (df_reduced[df_reduced['node'] == df_reduced['next_node']]
                 .rename(columns={'time':'arrival_time'})
                 .reset_index(drop=True))
    df_ends = (df_reduced[df_reduced['node'] == df_reduced['prev_node']]
               .rename(columns={'time':'depart_time'})
               .reset_index(drop=True))
    # now take all the pieces which have their indices reset and concat
    df_final = (pd.concat([df_starts['node'], df_ends['next_node'], df_starts['arrival_time'], 
                          df_ends['depart_time'], df_starts['index']], axis=1)
                .rename(columns={'next_node':'destination'}))
    
    # add in a time difference column.  cast to str because postgres doesnt like
    # pandas time intervals
    df_final['time_diff'] = (df_final['depart_time'] - df_final['arrival_time']).astype('str')
    
    # find the position count by subtracting the current index from the 
    # shifted index of the next row
    df_final['position_count'] = df_final['index'].shift(-1) - df_final['index']
    df_final.drop('index', axis=1, inplace=True)
    # add the mmsi we are working with to the pandas df
    df_final['mmsi'] = mmsi[0]
    # write back to the database
    df_final.to_sql(name=edge_table, con=loc_engine, if_exists='append',
                       method='multi', index=False )
    # increase the count
    run_count +=1
    # tracking will show how many mmsis have been processed and percentage complete.
    percentage = (run_count/len(mmsi_list_potential)) * 100
    print('Completed {} MMSIs.  {} percent complete.'.format(run_count, round(percentage,2)))

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

