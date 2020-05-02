import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import datetime

#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database="ais_test")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()

def create_sql_alch_engine(database):
    user = 'patrickmaus'
    host = 'localhost'
    port = '5432'
    return create_engine('postgresql://{}@{}:{}/{}'.format(user, host, 
                                                           port, database))
loc_engine = create_sql_alch_engine('ais_test')

#%% variable delcaration
port_activity_table = 'ship_position_ports'
edge_table = 'edges'
dist = 5
#%% Function for executing SQL
def execute_sql(SQL_string):
    c = conn.cursor()
    c.execute(SQL_string)
    conn.commit()
    c.close()
    
def port_check(row):
    if row['nearest_port_dist_km'] <=dist:
        val = row['nearest_port_id']
    else:
        val = 0
    return val


#%%
c = conn.cursor()
#c.execute("""DROP TABLE IF EXISTS {};""".format(edge_table))
conn.commit()
c.execute("""CREATE TABLE IF NOT EXISTS {}  (
        origin              int, 
        destination         int,
        mmsi                text, 
        depart_time         timestamp,
        position_count      bigint, 
        arrival_time        timestamp)         
""".format(edge_table))
conn.commit()
c.close()

#%% get mmsi
c = conn.cursor()
c.execute("""SELECT DISTINCT(mmsi) FROM {} 
          WHERE nearest_port_dist_km < {};"""
          .format(port_activity_table, dist))
mmsi_list_potential = c.fetchall()
c.close()

c = conn.cursor()
c.execute("""SELECT DISTINCT(mmsi) FROM {};"""
          .format(edge_table))
mmsi_list_completed = c.fetchall()
c.close()

# find the mmsis that are not in the edge table yet
diff = lambda l1,l2: [x for x in l1 if x not in l2]
mmsi_list = diff(mmsi_list_potential, mmsi_list_completed)
#%% Get the df from the database

first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

# define empty list
edge_list = []
mmsis_with_no_port = []

run_count = 0

for i in range(len(mmsi_list)): #iterate through all the mmsi #'s gathered
    # get the mmsi from the tuple
    mmsi = mmsi_list[i][0]
    
    print('Working on MMSI:', mmsi)
        
    # not efficent, but the easiest way to parse.  Largest number of positions
    # for all data is only about 40,000, so pandas can handle it
    df = pd.read_sql("""select time, nearest_port_id, nearest_port_dist_km 
                     from {} 
                     where mmsi = '{}' 
                     order by time"""
                     .format(port_activity_table, mmsi), loc_engine)
    
    # port check takes the dist to nearest port and if its less than dist, populates
    # port_id with the nearest port id.  If the nearest port is greater than dist,
    # port_id = 0.  0 will be used for activity "not in port"
    df['port_id'] = df.apply(port_check, axis=1)
  
    # define our key variables to start the iteration.  
    # Technically desitination and arrival time are not needed as they will
    # be defined in the iteration, but they need to be reset here when 
    # a new MMSI begins iteration.
    origin = 9999 # unused port_id to repersent undefined port
    depart_time = '2017-01-01T00:00:01' # could not insert a null
    destination = df['port_id'].iloc[0]
    arrival_time = df['time'].iloc[0] 
    position_count = 0
    
    for idx in (range(len(df)-1)):
        mmsi_edge_list = []
        
        # if the port id's are different, the ship went from one area to another
        if  df['port_id'].iloc[idx] != df['port_id'].iloc[idx+1]:
            
            # The ship went from open ocean to a port or from port to open ocean.
            # We don't want to log each one of these transitions.
            if df['port_id'].iloc[idx] == 0 or df['port_id'].iloc[idx+1] == 0:
                position_count += 1

            # This case is when the ports are different and do not include 0.
            # This is port-to-port activity we are looking for.
            else:
                destination = df['port_id'].iloc[idx]
                arrival_time = df['time'].iloc[idx]
                position_count += 1
               
                #  add to a list for easier debug
                mmsi_edge_list.append([origin, destination, mmsi, depart_time, 
                                        position_count, arrival_time])

                # Update the origin and depart for the next iteration.
                origin = df['port_id'].iloc[idx]    
                depart_time = df['time'].iloc[idx]
                position_count = 0
        
        #this case covers when a vessel does not make any changes
        elif df['port_id'].iloc[idx] == df['port_id'].iloc[idx+1]:
            position_count += 1
            
            # this doesnt handle ships that visit a port and then head off to sea.
            # need to finish.
            
        else: print ('something weird')
        
        # make a df from the mmsi_edge_list, push to sql, and extend to edge_list
        mmsi_df = pd.DataFrame(mmsi_edge_list, columns=('origin', 'destination', 'mmsi',
                                                   'depart_time', 'position_count', 
                                                   'arrival_time'))
        mmsi_df.to_sql(name=edge_table, con=loc_engine, if_exists='append',
                       method='multi', index=False )
        edge_list.extend(mmsi_edge_list)
        
    run_count +=1
    percentage = ((run_count+len(mmsi_list_completed))/len(mmsi_list_potential)) * 100
    print('Completed {} MMSIs.  {} percent complete.'.format(run_count, round(percentage,2)))

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

#%%

#     mmsi = '366981360'
#     # not efficent, but the easiest way to parse.  Largest number of positions
#     # for all data is only about 40,000, so pandas can handle it
#     df = pd.read_sql("""select time, nearest_port_id, nearest_port_dist_km 
#                      from {} 
#                      where mmsi = '{}' 
#                      order by time"""
#                      .format(port_activity_table, mmsi), conn)
# #%%    

# def f(row):
#     if row['A'] == row['B']:
#         val = 0
#     elif row['A'] > row['B']:
#         val = 1
#     else:
#         val = -1
#     return val

# def port_check(row):
#     if row['nearest_port_dist_km'] <=dist:
#         val = row['nearest_port_id']
#     else:
#         val = 0
#     return val
        
# df['port_id'] = df.apply(port_check, axis=1)
# #%%

# first_tick = datetime.datetime.now()
# print('Starting Processing at: ', first_tick.time())
# df['port_id'] = df.apply(port_check, axis=1)

# # define our key variables to start the iteration.  
# # Technically desitination and arrival time are not needed as they will
# # be defined in the iteration, but they need to be reset here when 
# # a new MMSI begins iteration.
# origin = 9999 # unused port_id to repersent undefined port
# depart_time = '2017-01-01T00:00:01' # could not insert a null
# destination = df['port_id'].iloc[0]
# arrival_time = df['time'].iloc[0] 
# position_count = 0

# print('Working on MMSI:', mmsi)

# for idx in (range(len(df)-1)):
#     mmsi_edge_list = []
    
#     # if the port id's are different, the ship went from one area to another
#     if  df['port_id'].iloc[idx] != df['port_id'].iloc[(idx+1)]:
        
#         # The ship went from open ocean to a port or from port to open ocean.
#         # We don't want to log each one of these transitions.
#         if df['port_id'].iloc[idx] == 0 or df['port_id'].iloc[idx+1] == 0:
#             position_count += 1

#         # This case is when the ports are different and do not include 0.
#         # This is port-to-port activity we are looking for.
#         else:
#             destination = df['port_id'].iloc[idx]
#             arrival_time = df['time'].iloc[idx]
#             position_count += 1

#             # add to the db
#             # insert_edge_sql = """INSERT INTO edges (origin, destination, 
#             # mmsi, depart_time, position_count, arrival_time) VALUES (%s,%s, %s, %s, %s, %s)"""
#             # record_values = (int(origin), int(destination), mmsi, depart_time, 
#             #                  int(position_count), arrival_time)         
#             # c = conn.cursor()
#             # c.execute(insert_edge_sql, record_values)
#             # conn.commit()
#             # c.close()

            
#             #  add to a list for easier debug
#             mmsi_edge_list.append([origin, destination, mmsi, depart_time, 
#                                     position_count, arrival_time])

#             # Update the origin and depart for the next iteration.
#             origin = df['port_id'].iloc[idx]    
#             depart_time = df['time'].iloc[idx]
#             position_count = 0
    
#     #this case covers when a vessel does not make any changes
#     elif df['port_id'].iloc[idx] == df['port_id'].iloc[idx+1]:
#         position_count += 1
        
#         # this doesnt handle ships that visit a port and then head off to sea.
#         # need to finish.
        
#     else: print ('something weird')
    
#     # make a df from the mmsi_edge_list, push to sql, and extend to edge_list
#     mmsi_df = pd.DataFrame(mmsi_edge_list, columns=('origin', 'destination', 'mmsi',
#                                                'depart_time', 'position_count', 
#                                                'arrival_time'))
    
# last_tock = datetime.datetime.now()
# lapse = last_tock - first_tick
# print('Processing Done.  Total time elapsed: ', lapse)
# #%%



