import numpy as np
from numba import jit

max_nodes = 100#55_000

max_neighbors = max_nodes-1

graph_array = np.zeros((max_nodes, max_neighbors), dtype=int)

neighbor_ids = np.arange(max_nodes)

neighbors = np.empty(max_nodes - 1, dtype=neighbor_ids.dtype)

@jit(nopython=True)
def node_creation(max_nodes, neighbor_ids, neighbors, graph_array):
    for n in range(max_nodes):
        neighbors[:n] = neighbor_ids[:n]      # view copy into buffer
        neighbors[n:] = neighbor_ids[n+1:]    # view copy into buffer
        
        graph_array[n, :max_nodes-1] = neighbors
        graph_array[neighbors, max_nodes-2] = n
        #print(n)
    return graph_array
#print(graph_array)


graph_array = node_creation(max_nodes, neighbor_ids, neighbors, graph_array)
np.save('graph_array_g3.npy', graph_array)