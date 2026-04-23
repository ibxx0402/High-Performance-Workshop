import random
import numpy as np
import numba


@numba.jit()
def make_graph(size,max_edges):
    #TODO Make this fast. it has a very high big O notation.


    # Fills up the graph with edges.
   # graph = [[-1]*max_edges]*size #The start.



    graph = np.zeros(shape=(size,max_edges),dtype=np.int32)-1
    
    connected_to_node = np.zeros(size,dtype=np.bool)
    full_node = np.zeros(size,dtype=np.bool)

    # G
    for n in range(0,size):

        if n == 0:
            connected_to_node[0] = True

        for i in range(0,max_edges):

            if graph[n][i] == -1:
             # if graph[n][i] == -1:

                # Select a random node not connected. 
                if connected_to_node[n] == False:

                    a = ~full_node & connected_to_node
                    a[n] = False #don't pick yourself.
                    a[np.where(graph[n] > -1) [0]] = False

                    x = np.random.choice(np.where(a == True)[0])
                    connected_to_node[n] = True # x is now connected

                # If it is connected select random unconnected.
                elif connected_to_node[n] == True and False in connected_to_node:

                    # not a full node and not connected to node.
                    a = ~full_node & ~connected_to_node
                    a[n] = False
                    a[np.where(graph[n] > -1) [0]] = False

                    x = np.random.choice(np.where(a == True)[0])
                    connected_to_node[x] = True
                else:
                    a = ~full_node
                    a[n] = False
                    a[np.where(graph[n] > -1) [0]] = False
                    try:
                        x = np.random.choice(np.where(a == True)[0])
                    except:
                        print("Not all edges can be used in this case, but all is connected.")
                        break

                #print(f"rabdin {x}"
                #if -1 in graph[x]:
                graph[n][i] = x
                t = np.where(graph[x] == -1)
                graph[x][t[0][0]] = n

                # TODO add logic Add connected to note and full node.

                if -1 not in graph[x]:
                    full_node[x] = True
                if -1 not in graph[n]:
                    full_node[n] = True
        if n % 10_000 == 0:
            print(f"Loop {n} done")
              
    return graph


if __name__ == "__main__":

    print(make_graph(100_000,3))