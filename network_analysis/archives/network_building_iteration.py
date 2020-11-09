import pandas as pd
import datetime

from multiprocessing import Pool

# Geo-Spatial Temporal Analysis package
import gsta
import db_config

conn = gsta.connect_psycopg2(db_config.loc_cargo_params)
loc_engine = gsta.connect_engine(db_config.loc_cargo_params)

#%% variable delcaration
nearest_site_table = 'nearest_site'
edge_table = 'cargo_edgelist_1km'
dist = 1

#%% port check function
def port_check(row, dist=dist):
    if row['nearest_site_dist_km'] <=dist:
        val = row['nearest_site_id']
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
        uid                text
        )         
""".format(edge_table))
conn.commit()
c.close()

#%% set indices for tables.

c = conn.cursor()
conn.commit()
c.execute(f"""CREATE INDEX {str(edge_table)}_uid_idx on {edge_table} (uid);""")
conn.commit()
c.close()


#%% get uid lists

# This list is all of the uids in the table of interest.  It is the
# total number of uids we will be iterating over.
c = conn.cursor()
c.execute("""SELECT DISTINCT(uid) FROM uid_positions;""")
uid_list_potential = c.fetchall()
c.close()

# if we have to stop the process, we can use the uids that are already completed
# to build a new list of uids left to complete.  this will allow us to resume
# processing without repeating any uids.
c = conn.cursor()
c.execute("""SELECT DISTINCT(uid) FROM {};"""
          .format(edge_table))
uid_list_completed = c.fetchall()
c.close()

# find the uids that are not in the edge table yet
diff = lambda l1,l2: [x for x in l1 if x not in l2]
uid_list = diff(uid_list_potential, uid_list_completed)
#%% iterate through the uid list and build the network edges
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

# run count = total uids already processed plus current batch
run_count = len(uid_list_completed)


for i in range(len(uid_list)): #iterate through all the uid #'s gathered
    # get the uid from the tuple
    uid = uid_list[i]

    print('Working on uid:', uid[0])

    # not efficent, but the easiest way to parse.  Largest number of positions
    # for all data is only about 50,000, so pandas can handle it
    df = pd.read_sql(f"""SELECT s.id, s.nearest_site_id, s.nearest_site_dist_km,
                    pos.time, pos.uid, pos.lat, pos.lon
                    FROM nearest_site AS s, uid_positions AS pos
                    where s.id = pos.id
                    AND pos.uid = '{uid[0]}'
                    ORDER BY TIME;""", loc_engine)


    # port_check takes the dist to nearest port and if its less than dist, populates
    # port_id with the nearest port id.  If the nearest port is greater than dist,
    # port_id = 0.  0 will be used for activity "not in port"
    df['node'] = df.apply(port_check, axis=1)
    # no longer need port_id and dist
    df.drop(['nearest_site_id', 'nearest_site_dist_km'], axis=1, inplace=True)
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
    # add the uid we are working with to the pandas df
    df_final['uid'] = uid[0]
    # write back to the database
    df_final.to_sql(name=edge_table, con=loc_engine, if_exists='append',
                       method='multi', index=False )
    # increase the count
    run_count +=1
    # tracking will show how many uids have been processed and percentage complete.
    percentage = (run_count/len(uid_list_potential)) * 100
    print('Completed {} uids.  {} percent complete.'.format(run_count, round(percentage,2)))

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

