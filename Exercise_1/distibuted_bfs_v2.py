from mpi4py import MPI
import numpy as np

# mpiexec -n 3 python Exercise_1\distibuted_bfs.py

# Tag index
# Tag 1 = Sending initial graph parts.
# tag 2 = Sending a vertices to check out.


def bfsv_parallel(graph, source):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # often just zero, seems to be "random"
    world_size = comm.Get_size() # How many processes aviable

    # TODO Split up the array.

  

    
    # Master p is rank 0
    if rank == 0:
        V = len(graph)

        assert V % world_size == 0, f"{V} % {world_size} = {V % world_size} Error the number of processors must be a mutliple of number of vertices"

        distance = np.full(V, -1, dtype=np.int32)
        distance[source] = 0

        frontier_array = np.zeros(V,dtype=np.bool)
        frontier_array[source] = True

        split = len(graph)//world_size
        split_offset = 0
        if len(graph)%world_size != 0: # otherwise data for another processor is made.
            print()
            split+=1
            split_offset = 1


        print(split+split_offset)
        node_splits = np.full((world_size,2),-1,dtype=np.int32)
        print(node_splits)
        print(f"world size = {world_size}")
        for i in range(0,world_size):

            node_splits[i,0]= i*split
            node_splits[i,1] = (i+1)*split-1



            if rank == 0 and i == 0:
                lo_nodes = graph[i*split:(i+1)*split]
                lo_distance = distance[i*split:(i+1)*split]
                lo_frontier_array = frontier_array[i*split:(i+1)*split]

            elif rank == 0 and i != 0:
                vertex_tx = graph[i*split:(i+1)*split]
                #print(vertex_tx)
                distance_split_tx = distance[i*split:(i+1)*split]
                frontier_array_tx = frontier_array[i*split:(i+1)*split]


                print(f"p{rank} sends data to rank {i}")
                comm.send([vertex_tx,distance_split_tx,frontier_array_tx,node_splits],dest=i,tag=1) # Tag 1 = 
                pass

    if rank != 0:
        # graph should in princible also be send, at least their part of it.
        lo_nodes,lo_distance,lo_frontier_array,node_splits = comm.recv(tag = 1)
        distance = np.zeros(len(graph))
        #print(f"p{rank} {lo_frontier_array}")

    V = len(lo_nodes)
    frontier = 1
    

    #if rank == 0:
        #print(f"node splits = {node_splits}")

    while True:
        lo_neighbor_array = np.zeros(V, dtype=np.bool_) # The Visited nodes in this itteration
        lo_to_send = np.full(len(lo_nodes[0])*len(lo_nodes),-1)
        full_list = np.full(len(graph)*len(graph[0]),-1)

        #if rank == 1:
            #print(f"p1 lo frontier = {lo_frontier_array}")

        for u in range(V): #Can be length V instead as every FoundS is V length
            
            # Offset
            if lo_frontier_array[u]: # if the node is a frontier array.
                for v in lo_nodes[u]:
                    #print(f"p{rank} {node_splits[rank,0]} <= {v} <= {node_splits[rank,1]}")
                    if node_splits[rank,0] <= v and v <= node_splits[rank,1]:
                        
                        v_a = index_offset(v,node_splits,rank)
                        if lo_distance[v_a] == -1:
                            lo_neighbor_array[v_a] = True      
                            lo_distance[v_a] = frontier
                    elif v != -1:
                        # TODO implement send to all logic while recieving at the same time.
                        

                        tmp = np.where(lo_to_send == -1)
                        #print(f"p{rank} tmp = {tmp} tmp[0] = {tmp[0]}, {lo_to_send}")
                        if v not in lo_to_send:
                            #print(f"p{rank} wil send {v} while frontier = {frontier}")
                            lo_to_send[tmp[0][0]] = v

                        pass

        #if rank == 0:
            #print(f"p{rank} frontier {frontier} tosend = {lo_to_send}")

        # TODO Here we send the estra we need to send.

        comm.barrier

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


        #if rank == 1:
            #print(f"p1 lo neighbor = {lo_neighbor_array}")
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

        # TODO Send adjacent neigbors to other processes

        #if rank == 1:
            #print(f"p{rank} revieced = {full_list}")
        
        #break

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

    # works with processors = 3 and the graph

    # TODO Make a proper logic to end the loop.

    graph = np.array([[1,2,4],[3,4,-1],[4,5,-1],[6,-1,-1],[6,7,8],[7,-1,-1],[8,-1,-1],[8,-1,-1],[-1,-1,-1]],dtype=np.int32)

    # Nasty graphs not supported. len of graph must be multiple of number of processes
    nasty_graph = np.array([[1,2,4],[3,4,-1],[4,5,-1],[6,-1,-1],[6,7,8],[7,-1,-1],[8,-1,-1],[8,-1,-1],[-1,-1,9],[10,-1,-1],[11,-1,-1],[-1,-1,-1]],dtype=np.int32)
    source = 0

    t = bfsv_parallel(nasty_graph,source)


    