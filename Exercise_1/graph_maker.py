import random
import numpy as np

def make_graph(size,max_edges):


    # Fills up the graph with edges.
   # graph = [[-1]*max_edges]*size #The start.



    graph = np.zeros(shape=(size,max_edges),dtype=np.int32)-1

    

    # G
    for n in range(0,size):


        for i in range(0,max_edges):


           # if graph[n][i] == -1:

                # Select a random node not connected. 
            x = random.choice([i for i in range(0,size) if i != n and i not in graph[n]]) 
         
            #print(f"rabdin {x}")



            graph[n][i] = x
             

              



    return graph


print(make_graph(10,3))