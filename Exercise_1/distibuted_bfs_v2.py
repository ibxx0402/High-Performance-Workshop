from mpi4py import MPI
import numpy as np

# mpiexec -n 3 python Exercise_1\distibuted_bfs.py




def bfsv_dist_mem_parallel(graph, source):
    """
    Breath First seach with dis

    Only works if the number of vertices (nodes) is a multiple of the numer of processors.
    
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # often just zero, seems to be "random"
    world_size = comm.Get_size() # How many processes aviable

    # Master p is rank 0
    if rank == 0:
        V = len(graph)

        assert V % world_size == 0, f"{V} % {world_size} = {V % world_size} Error the number of processors must be a mutliple of number of vertices"

        distance = np.full(V, -1, dtype=np.int32)
        distance[source] = 0

        frontier_array = np.zeros(V,dtype=np.bool)
        frontier_array[source] = True

        split = len(graph)//world_size
        #split_offset = 0

        # As we only support multiples, the distribution is always equal, so this does nothing.
        if len(graph)%world_size != 0: # otherwise data for another processor is made. 
            split+=1
            #split_offset = 1


        node_splits = np.full((world_size,2),-1,dtype=np.int32)

        for i in range(0,world_size):

            node_splits[i,0]= i*split
            node_splits[i,1] = (i+1)*split-1

            # local variables for process 0
            if rank == 0 and i == 0:
                lo_nodes = graph[i*split:(i+1)*split]
                lo_distance = distance[i*split:(i+1)*split]
                lo_frontier_array = frontier_array[i*split:(i+1)*split]

            # send local variables to the others.
            elif rank == 0 and i != 0:
                vertex_tx = graph[i*split:(i+1)*split]
                distance_split_tx = distance[i*split:(i+1)*split]
                frontier_array_tx = frontier_array[i*split:(i+1)*split]
                comm.send([vertex_tx,distance_split_tx,frontier_array_tx,node_splits],dest=i,tag=1) # Tag 1 = 
                pass

    if rank != 0:
        # graph should in princible also be send, at least their part of it.
        lo_nodes,lo_distance,lo_frontier_array,node_splits = comm.recv(tag = 1)
        distance = np.zeros(len(graph))

    V = len(lo_nodes)
    frontier = 1

    while True:
        lo_neighbor_array = np.zeros(V, dtype=np.bool_) # The Visited nodes in this itteration

        # Worst case is that every neighbor for every vertex points to other processers vertices, so the lo_to_send buffer needs to be big.
        lo_to_send = np.full(len(lo_nodes[0])*len(lo_nodes),-1)
        full_list = np.full(len(graph)*len(graph[0]),-1)

        # Look for edges in each local vertic
        for u in range(V):
            
            if lo_frontier_array[u]: # if the node is a frontier array.
                for v in lo_nodes[u]:
                    if node_splits[rank,0] <= v and v <= node_splits[rank,1]:
                        
                        # Adjust the node ID to fit into the local lo_node array
                        v_a = index_offset(v,node_splits,rank)
                        if lo_distance[v_a] == -1:
                            lo_neighbor_array[v_a] = True      
                            lo_distance[v_a] = frontier

                    # if vertix v is not in local node, send it to other processers.
                    elif v != -1:
                        tmp = np.where(lo_to_send == -1)

                        # Find an empty spot for it. (THis should be a queue, would be less messy)
                        if v not in lo_to_send:
                            lo_to_send[tmp[0][0]] = v

                        pass
        comm.barrier()

        # A bit of overhead as np.list are send.
        comm.Allgather(lo_to_send,full_list)
    
        for v in full_list:
            if v == -1:
                pass
            else:
                if node_splits[rank,0] <= v and v <= node_splits[rank,1]:
                        v_a = index_offset(v,node_splits,rank)
                        try:
                            if lo_distance[v_a] == -1:
                                lo_neighbor_array[v_a] = True     
                                lo_distance[v_a] = frontier
                        except:
                            print(f"p{rank} error")
                            print(f"p{rank} uses node {v} got changed to {v_a} and it is not found in {lo_distance}")



        lo_frontier_array = lo_neighbor_array # The now visited notes
        frontier += 1

        #if rank == 1:
            #print(f"p{rank} lo_neighbor = {lo_neighbor_array}")

        comm.barrier()

        front_buffer = np.zeros(len(graph),dtype=np.bool)

        # a bit of overhad at 
        comm.Allgather(lo_frontier_array,front_buffer)

        if not np.any(front_buffer):
            break
 

        comm.barrier()

    comm.barrier()

    final_result = np.zeros(len(graph),dtype=np.int32)

    comm.Gather(lo_distance,final_result,root = 0)
    if rank == 0:
        print(f"final result = {final_result}")

        return final_result






def index_offset(i,split,rank):
    """
    Adjust the node_id to local index
    """

    adjusted = i-split[rank,0]

    return adjusted





if __name__ == "__main__":


    graph = np.array([[1,2,4],[3,4,-1],[4,5,-1],[6,-1,-1],[6,7,8],[7,-1,-1],[8,-1,-1],[8,-1,-1],[-1,-1,-1]],dtype=np.int32)

    nasty_graph = np.array([[1,2,4],[3,4,-1],[4,5,-1],[6,-1,-1],[6,7,8],[7,-1,-1],[8,-1,-1],[8,-1,-1],[-1,-1,9],[10,-1,-1],[11,-1,-1],[-1,-1,-1]],dtype=np.int32)
    source = 0

    super_graph = np.load("graph_array_random.npy")



    t = bfsv_dist_mem_parallel(super_graph,source)

    from exercise_1_a import bfsv_python

    x = bfsv_python(super_graph,source)

    print(f"t and x is the same = {np.array_equal(t,x)}")



    