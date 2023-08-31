"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
##################

G = nx.read_edgelist('../datasets/CA-HepTh.txt', comments='#', delimiter='\t')
print(f'The graph has {len(list(G.nodes))} nodes.')
print(f'The graph has {len(list(G.edges))} edges.')

############## Task 2

##################
# your code here #
##################

ccs = list(nx.connected_components(G))
print(f'The graph has {len(ccs)} connected components.')

gcc = G.subgraph(ccs[0])
print(f'The giant connected component has {len(list(gcc.nodes))} nodes and {len(list(gcc.edges))} edges.')
print(f'This represents a ratio of {len(list(gcc.nodes))/len(list(G.nodes)):.2f} of all the nodes.')
print(f'This represents a ratio of {len(list(gcc.edges))/len(list(G.edges)):.2f} of all the edges.')


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################
print(f'The minimum degree of a vertex is {np.min(degree_sequence)}')
print(f'The maximum degree of a vertex is {np.max(degree_sequence)}')
print(f'The mean degree of a vertex is {np.mean(degree_sequence)}')

degree_sequence.sort()
print(f'The median degree of a vertex is {degree_sequence[(len(degree_sequence)-1)//2]}')


############## Task 4

##################
# your code here #
##################

plt.hist(degree_sequence, bins=np.max(degree_sequence) - np.min(degree_sequence) + 1, label='Degree distribution over the vertices')
plt.legend()
plt.show()

plt.hist(degree_sequence, bins=np.max(degree_sequence) - np.min(degree_sequence) + 1, label='Log-log degree distribution over the vertices', log=True)
plt.xscale('log')
plt.legend()
plt.show()
############## Task 5

##################
# your code here #
##################

print(f'The transitivity of the graph is {nx.transitivity(G)}')