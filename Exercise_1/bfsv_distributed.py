import numpy as np 
from mpi4py import MPI



def bfsv_distributed(source):
  
    #INIT MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    if rank == 0:
        graph = np.load('graph_array_example.npy')
        V = len(graph)
        chunk = V // size
        chunks = [graph[i * chunk:(i + 1) * chunk] for i in range(size)]
    else:
        V = None
        chunks = None

    V = comm.bcast(V, root=0) 
    frontier_stack = comm.scatter(chunks, root=0)

    #Init bfs 
    distance = np.full(V, -1, dtype=np.int32)
    distance[source] = 0
    frontier = 0
 
    neighbor_stack = []
    print(f"Rank {rank} frontier_stack {frontier_stack} test")
    
    while True:
        break
        #all vertices traversed
        """ if frontier_stack == [] for all processors then:
            return distance """


    """ #begin BFS traversal
    while True:
        local_start = rank * chunk
        local_end   = local_start + chunk
        frontier_stack = graph[local_start: local_end]

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
            d[v] = level + 1  """
    
bfsv_distributed(0)