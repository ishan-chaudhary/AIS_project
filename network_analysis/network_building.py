import psycopg2
import pandas as pd
import datetime
import matplotlib as plt
import networkx as nx

#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database="ais_test")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()

#%% Function for executing SQL
def execute_sql(SQL_string):
    c = conn.cursor()
    c.execute(SQL_string)
    conn.commit()
    c.close()

#%%
c = conn.cursor()
c.execute("""DROP TABLE IF EXISTS edges_sample;""")
conn.commit()
c.execute("""CREATE TABLE IF NOT EXISTS edges_sample  (
        origin              int, 
        destination         int,
        mmsi                text, 
        depart_time         timestamp,
        position_count      bigint, 
        arrival_time        timestamp)         
""")
conn.commit()
c.close()

#%% get mmsi
c = conn.cursor()
c.execute("""SELECT DISTINCT(mmsi) FROM ship_position_sample;""")
mmsi_list = c.fetchall()
c.close()
#%% Get the df from the database

first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

# define empty list
edge_list = []


for i in range(len(mmsi_list)): #iterate through all the mmsi #'s gathered
    # get the mmsi from the tuple
    mmsi = mmsi_list[i][0]
    # not efficent, but the easiest way to parse.  Largest number of positions
    # for all data is only about 40,000, so pandas can handle it
    df = pd.read_sql("""select * from port_activity_sample 
                      where mmsi = '{}' order by time""".format(mmsi), conn)
    # networkx is using numeric nodes, so we will repersent at sea as 0
    df.fillna(value=0, inplace=True)  
    
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

                # add to the db
                insert_edge_sql = """INSERT INTO edges_sample (origin, destination, 
                mmsi, depart_time, position_count, arrival_time) VALUES (%s,%s, %s, %s, %s, %s)"""
                record_values = (int(origin), int(destination), mmsi, depart_time, 
                                 int(position_count), arrival_time)         
                c = conn.cursor()
                c.execute(insert_edge_sql, record_values)
                conn.commit()
                c.close()
                
                 # add to a list for easier debug
                #edge_list.append([idx, origin, destination, mmsi, depart_time, 
                #                        position_count, arrival_time])
                edge_list.append([origin, destination])

                # Update the origin and depart for the next iteration.
                origin = df['port_id'].iloc[idx]    
                depart_time = df['time'].iloc[idx]
                position_count = 0
            
        elif df['port_id'].iloc[idx] == df['port_id'].iloc[idx+1]:
            position_count += 1
            
        else: print ('something weird')

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

#%%
G = nx.MultiDiGraph()
G.add_edges_from(edge_list)

#%%
nx.draw(G)
nx.draw_networkx(G)