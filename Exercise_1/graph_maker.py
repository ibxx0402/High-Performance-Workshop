import random
import numpy as np

def make_graph(size,max_edges):
    #TODO Make this fast. it has a very high big O notation.


    # Fills up the graph with edges.
   # graph = [[-1]*max_edges]*size #The start.



    graph = np.zeros(shape=(size,max_edges),dtype=np.int32)-1
    connected_to_node = []
    unconnected = [i for i in range(0,size)]
    full_node = []

    # G
    for n in range(0,size):

        if n == 0:
            connected_to_node.append(0)
    

            unconnected.remove(n)


        for i in range(0,max_edges):


            if graph[n][i] == -1:
             # if graph[n][i] == -1:



                # Select a random node not connected. 
                if n in unconnected:
                    x = random.choice([k for k in range(0,size) if k in connected_to_node and k not in graph[n] and k != n and k not in full_node])
                elif n in connected_to_node and len(unconnected) > 0:
                    x = random.choice([k for k in range(0,size) if k in unconnected and k not in graph[n] and k != n and k not in full_node]) 
                else:
                    try:
                        x = random.choice([k for k in range(0,size) if k not in graph[n] and k != n and k not in full_node])
                    except:
                        print("Not all edges can be used in this case, but all is connected.")

                #print(f"rabdin {x}")

                if -1 in graph[x]:
                    graph[n][i] = x
                    t = np.where(graph[x] == -1)
                    graph[x][t[0][0]] = n
                else:
                    #graph[n][i] = x
                    pass

                # Not perfect to find if all is connected.
                if (x in unconnected and n in connected_to_node):
                    connected_to_node.append(x)
                    unconnected.remove(x)
                elif (n in unconnected and x in connected_to_node):
                    connected_to_node.append(n)
                    unconnected.remove(n)

                if -1 not in graph[x] and x not in full_node:
                    full_node.append(x)

            if -1 not in graph[n] and n not in full_node:
                full_node.append(n)
              

    print(full_node)
    print(connected_to_node)
    print(unconnected)
    return graph


if __name__ == "__main__":

    print(make_graph(10,3))