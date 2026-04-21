import numpy as np

max_nodes = 300_000
max_neighbors = 1000
graph_array = np.zeros((max_nodes, max_neighbors), dtype=int)

for n in range(max_nodes):
    neighbor_ids = list(range(max(0, n - max_neighbors), n))

    graph_array[n, :len(neighbor_ids)] = neighbor_ids

    graph_array[neighbor_ids, len(neighbor_ids)-1] = n

print(graph_array[0])
#np.save('graph_array_g2.npy', graph_array)