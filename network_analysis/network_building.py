import psycopg2
import pandas as pd
import matplotlib as plt
import networkx as nx

#%% Make and test conn and cursor
conn = psycopg2.connect(host="localhost",database="ais_data")
c = conn.cursor()
if c:
    print('Connection is good.')
c.close()

#%% Get the df from the database
port_df = pd.read_sql('select * from port_activity_reduced', conn)

#%% restructure the df to be ingested by networkx.  
# Each row needs to be start node, end node, and then attributes


#%%


port_list = []
for idx in range(len(port_df)):
    try:
        origin = port_df.iloc[idx, 4]
        destination = port_df.iloc[idx + 1, 4]
        depart_time = port_df.iloc[idx, 1]
        arrival_time = port_df.iloc[idx + 1, 0]
        
        port_list.append((origin, destination, {'depart_time': depart_time}))
        
    except: 
        print(idx)
        continue

#%%

G = nx.MultiDiGraph()
G.add_edges_from(port_list)




#%%
for k, v in G.degree():
    print (k,v)



#%%
#nx.draw(G)
nx.draw_networkx(G)