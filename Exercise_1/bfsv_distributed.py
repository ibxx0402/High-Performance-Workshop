import numpy as np 
from mpi4py import MPI

def bfsv_distributed(graph, source):
  
    #INIT MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    proc_count = comm.Get_size()



    #Init bfs 
    V = len(graph)
    print(V)
    distance = np.full(V, -1, dtype=np.int32)
    
    distance[source] = 0
    frontier = 0
    frontier_stack = []
    neighbor_stack = []

    split_graph_size = V / proc_count

    #begin BFS traversal
    while True:
        frontier_stack = {the set of local vertices with frontier}
        
        #all vertices traversed
        if frontier_stack == [] for all processors then:
            return distance

        #construct the NS based on local vertices in current frontier
        NS = {neighbors of vertices in FS, both local and not local vertices}
        #synchronization: all-to-all communication

        for j in range(proc_count):
            N_j = {vertices in NS owned by processor j}
            send N_j to processor j
            receive N_j_rcv from processor j


        #combine the received message to form local next vertex frontier then update the level for them
        NS_rcv = Union(N_j_rcv)
        for v in NS_rcv and d[v] == -1 do
            d[v] = level + 1
    
Example_graph = np.load('graph_array_example.npy')
print(bfsv_distributed(Example_graph, 0))