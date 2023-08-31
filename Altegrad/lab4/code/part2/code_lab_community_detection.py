"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    # your code here #
    ##################
    
    #Step 1
    A = nx.adjacency_matrix(G)

    #Step 2
    D = diags([1/G.degree(node) for node in G.nodes()])
    Lrw = eye(A.shape[0]) - D @ A

    #Step 3
    _, U = eigs(Lrw, k=k, which='SM')
    U = U.real

    #Step 4
    kmeans = KMeans(n_clusters=k).fit(U)

    clustering = {node:kmeans.predict(U[i].reshape(1,-1))[0] for i, node in enumerate(list(G.nodes()))}
    return clustering





############## Task 7

##################
# your code here #
##################

G = nx.read_edgelist('../datasets/CA-HepTh.txt', comments='#', delimiter='\t')
gcc = G.subgraph(list(nx.connected_components(G))[0])
clustering = spectral_clustering(gcc, 50)

############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    clusters = {}
    for node, cluster in clustering.items():
        if cluster in clusters.keys():
            clusters[cluster].append(node)
        else:
            clusters[cluster] = [node]
    modularity = 0
    m = G.number_of_edges()
    for _, cluster in clusters.items():
        g = nx.subgraph(G, cluster)
        modularity += (g.number_of_edges()/m) - (np.sum([G.degree(node) for node in g.nodes()])/(2*m))**2
    
    return modularity



############## Task 9

##################
# your code here #
##################

modularity1 = modularity(G, clustering)
random_clustering = {node: randint(1, 50) for node in G.nodes()}
modularity2 = modularity(G, random_clustering)

print(f'The modularity of the CA-HepTh with spectral clustering is {modularity1}')
print(f'The modularity of the CA-HepTh with random clustering is {modularity2}')





