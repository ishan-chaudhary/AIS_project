import psycopg2
import pandas as pd
import matplotlib as plt
import networkx as nx

#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database="ais_test")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()

#%% Get the df from the database
df = pd.read_sql("""select * from port_activity_sample 
                      where mmsi = '212917000'""", conn)
df.fillna(value=0, inplace=True)
df.drop('geog', axis=1, inplace=True)

#%% restructure the df to be ingested by networkx.  
# Each row needs to be start node, end node, and then attributes
edge_list = []
attr_list = []
position_count = 0
origin = df['port_id'].iloc[0]
depart_time = df['time'].iloc[0]

for idx in (range(len(df)-1)):
    if df['port_id'].iloc[idx] == df['port_id'].iloc[idx+1]:
        position_count += 1
      
    else:
        destination = df['port_id'].iloc[idx+1]
        arrival_time = df['time'].iloc[idx+1]
        if position_count > 10:
            edge_list.append([origin, destination])
            attr_list.append((origin, destination,
                          [{'depart_time': depart_time},
                          {'arrival_time' : arrival_time}, 
                          {'position_count' : position_count}]))
            position_count = 0
            origin = df['port_id'].iloc[idx+1]
            depart_time = df['time'].iloc[idx]
            
        else: continue

#%%

G = nx.MultiDiGraph()
G.add_edges_from(edge_list)




#%%
#nx.draw(G)
nx.draw_networkx(G)