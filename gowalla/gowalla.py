import pandas as pd
import matplotlib.pyplot as plt
import gsta
import gsta_config

# %%
df_edges = pd.read_csv('gowalla/Gowalla_edges.txt', sep='\t', header=None, dtype=str)
df_edges.columns = ['Source', 'Target']

# %%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
loc_engine = gsta.connect_engine(gsta_config.gowalla_params)
# %%
c = conn.cursor()
c.execute("""CREATE TABLE if not exists checkins (
  	uid 			text,
    time     		timestamp,
	lat				numeric,
	lon				numeric,
	loc_id          text
);""")
conn.commit()
# %%
generator = pd.read_csv('gowalla/Gowalla_totalCheckins.txt', sep='\t', header=None,
                        names=['uid', 'time', 'lat', 'lon', 'loc_id'], chunksize=100000)
for df in generator:
    df.to_sql(name='checkins', con=loc_engine, if_exists='append', method='multi', index=False)

# %%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
c.execute("""ALTER TABLE checkins
add column geog geography(Point, 4326);""")
conn.commit()
c.execute("""
UPDATE checkins SET geog = ST_SetSRID(
	ST_MakePoint(lon, lat), 4326);""")
conn.commit()
c.close()
conn.close()
#%% make a sites table
def make_sites(new_table_name, source_table, conn):
    c = conn.cursor()
    c.execute(f"""DROP TABLE IF EXISTS {new_table_name};""")
    conn.commit()
    c.execute(f"""CREATE TABLE {new_table_name} AS
    SELECT 
        loc_id,
        position_count,
        unique_uids,
        lat,
        lon,
        geom
        FROM (  
                SELECT pos.loc_id as loc_id,
                COUNT (pos.geom) as position_count,
                COUNT (DISTINCT (pos.uid)) as unique_uids,
                pos.lat as lat,
                pos.lon as lon,
                pos.geom as geom
                FROM {source_table} as pos
                GROUP BY pos.loc_id, pos.lat, pos.lon, pos.geom) 
                AS foo;""")
    conn.commit()
    c.execute(f"""CREATE INDEX if not exists sites_loc_id_idx on {new_table_name} (loc_id);""")
    conn.commit()
    c.close()


conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
make_sites('sites', 'checkins', conn)
conn.close()
#%% make trips table
def make_trips(new_table_name, source_table, conn):
    c = conn.cursor()
    c.execute(f"""DROP TABLE IF EXISTS {new_table_name};""")
    conn.commit()
    c.execute(f"""CREATE TABLE {new_table_name} AS
    SELECT 
        uid,
        position_count,
        first_date,
        last_date,
        last_date - first_date as time_diff,
        line
        FROM (
                SELECT pos.uid as uid,
                COUNT (pos.geog) as position_count,
                ST_MakeLine(pos.geog::geometry ORDER BY pos.time) AS line,
                MIN (pos.time) as first_date,
                MAX (pos.time) as last_date
                FROM {source_table} as pos
                GROUP BY pos.uid) 
                AS foo
        WHERE position_count > 2;""")
    conn.commit()
    c.execute(f"""CREATE INDEX if not exists trips_uid_idx on {new_table_name} (uid);""")
    conn.commit()
    c.close()


conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
make_trips('trips', 'checkins', conn)
conn.close()
#%%
loc_engine = gsta.connect_engine(gsta_config.gowalla_params)
df = pd.read_sql(f"""select uid, loc_id
                 from checkins
                 where uid = '{0}'
                 order by time""", loc_engine)

df.columns = ['uid', 'Source']
df['Target'] = df['Source'].shift(-1)
#%%
df_edgelist_weighted = (df.groupby(['Source', 'Target'])
                        .count()
                        .rename(columns={'uid': 'weight'})
                        .reset_index())

#%%
import networkx as nx
G = nx.from_pandas_edgelist(df, source='Source',
                            target='Target', create_using=nx.DiGraph)

df_report = pd.DataFrame([list(G),
                          [len(G[node]) for node in G.nodes()],
                          list(nx.degree_centrality(G).values()),
                          nx.in_degree_centrality(G).values(),
                          nx.out_degree_centrality(G).values(),
                          nx.eigenvector_centrality(G).values(),
                          nx.closeness_centrality(G).values(),
                          nx.betweenness_centrality(G).values()]
                         ).T
df_report.columns = ['Node', 'Targets', 'Degree', 'In-Degree', 'Out-Degree',
                     'Eigenvector', 'Centrality', 'Betweenness']
df_report = (df_report.astype({'Degree':'float', 'In-Degree':'float', 'Out-Degree':'float',
                               'Eigenvector':'float', 'Centrality':'float', 'Targets': 'int',
                               'Betweenness':'float'}).round(3))
df_report.hist()
plt.show()

df_report.boxplot(['Degree', 'In-Degree', 'Out-Degree', 'Eigenvector', 'Centrality',
                  'Betweenness'])
plt.show()

print(df_report.sort_values('Betweenness'))


#%%
# define the positions here so all the cluster plots in the loop are the same structure
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
plt.axis('off')
plt.title('Full Network Plot')
plt.show()

#%%
df_analysis = pd.read_sql(f"""SELECT 
s.loc_id,
s.position_count, 
s.unique_uids
c.uid, 
count(c.loc_id) as uid_counts
FROM sites as s, checkins as c
WHERE s.loc_id = '420315' and
c.uid = '0';
""", loc_engine)




#%%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
c.execute("""SELECT DISTINCT(uid) FROM trips;""")
uid_list = c.fetchall()
c.close()
conn.close()


#%%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
for uid in uid_list:
    try:
        c.execute(f"""select ST_Length(geography(line))/1000 AS line_length_km
                    from trips
                    where uid = '{uid[0]}';""")

    except: print('Failed:', uid)
conn.close()
#%%
conn = gsta.connect_psycopg2(gsta_config.gowalla_params)
c = conn.cursor()
c.execute("""select ST_Length(geography(line))/1000 AS line_length_km
            from trips
            where uid = '9999';""")
print(c.fetchone())
conn.close()