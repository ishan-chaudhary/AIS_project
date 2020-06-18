import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
# plotting
import matplotlib.pyplot as plt
df_edgelist_weighted = pd.read_csv('edgelist_weighted.csv')


# %% Plot the whole network
plt.figure(figsize=(10, 10))
G = nx.from_pandas_edgelist(df_edgelist_weighted, source='Source',
                            target='Target', edge_attr=True,
                            create_using=nx.DiGraph)

edges = G.edges()
weights = [np.log((G[u][v]['weight'])+.1) for u, v in edges]
pos = nx.spring_layout(G)  # positions for all nodes
# nodes
nx.draw_networkx_nodes(G, pos)
# edges
nx.draw_networkx_edges(G, pos, width=weights)
# labels
nx.draw_networkx_labels(G, pos)
plt.axis('off')
plt.title('Full Network Plot')
plt.show()

# %% Build Markov chain
markov = {}
for port in G.nodes:
    total = 0
    port_markov = {}
    for n in G[port]:
        total += (G[port][n]['weight'])
    for n in G[port]:
        port_markov[n] = (G[port][n]['weight'] / total)
    # make sure only nodes which have connections to other nodes are added
    if total > 0:
        markov[port] = port_markov

print(markov)
df_markov = pd.DataFrame(markov)


#%% calculate the number of hops between two ports

# assign the port you start chain at and target port
first_port = df_edgelist_weighted['Source'].sample().values[0]
target_port = df_edgelist_weighted['Source'].sample().values[0]

# first_port = 'BOSTON'
# target_port = 'MIAMI'

# target_hop_counter will store how many hops it took to reach the target port in each run
target_hop_counter = list()
# number of runs
run_target = 10000
# run tracker
run_counter = 0
# max run for entire loop, including dead ends
max_runs = run_target*5
# max run for counter for an individual chain
max_iterations = 1000



while len(target_hop_counter) < run_target:
    # at each round, the starting port needs to be re-set
    start_port = first_port
    # the port chain stores the ports visited in each run
    port_chain = list()
    # start each chain by adding the first port.
    port_chain.append(first_port)

    # pass the start_port to the markov dict to access the dictionary holding the markov
    # state information for that port.  Ports visited from this port are the keys, and
    # the proportion of times traveled to each port are the values.  This if statement
    # checks to make sure the start_port has at least one port that has been traveled to
    # from the start_port
    # use the markov state information to randomly select a port based on the
    # past observances
    next_port = np.random.choice(list(markov[start_port].keys()),
                                 p=list(markov[start_port].values()))
    # append that port to the chain
    port_chain.append(next_port)
    # set the next_port as the start_port to continue the chain
    start_port = next_port

    # counter will keep track of iterations within port chain
    iteration_counter = 0

    while next_port != target_port:
        try:
            # use the markov state information to randomly select a port based on the
            # past observances
            next_port = np.random.choice(list(markov[start_port].keys()),
                                         p=list(markov[start_port].values()))
            # append that port to the chain
            port_chain.append(next_port)
            # set the next_port as the start_port to continue the chain
            start_port = next_port

            # check if the iteration counter for the port chain is exceeded
            iteration_counter = iteration_counter + 1
            if iteration_counter < max_iterations:
                continue
            else:
                print('Max iterations exceeded.  Breaking.')
                break
        except Exception:
            # port is a dead end.  break the function
            break

    # only if the next port is the target port, add the length to the counter
    if next_port == target_port:
        target_hop_counter.append(len(port_chain))

    # iterate total runs by 1
    run_counter += 1
    # check if the max run number is exceeded
    if run_counter < max_runs:
        continue
    else:
        print('Max runs exceeded.  Breaking.')
        break

print(target_hop_counter)
print(len(target_hop_counter)/run_counter)

# plot the counts it took to get to the target port
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(target_hop_counter, bins=100)
plt.title(f"Distribution of {len(target_hop_counter)} runs from {first_port.title()} to {target_port.title()}")
plt.show()

print('The mean is', np.mean(target_hop_counter))
print('The median is', np.median(target_hop_counter))
print('The mode is', stats.mode(target_hop_counter)[0][0])
