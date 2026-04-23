#Creation of a adjecency matrix for a grapth
# Created from a list of, containing a lists of edges between vertices, each index of the outer list represents a vertex. The inner list contains the index of the vertices.


""" vertex0 = [1, 2, 3]
vertex1 = [0]
vertex2 = [0, 3]
vertex3 = [0, 2]
Example_graph = [vertex0, vertex1, vertex2, vertex3]  """

"""
link to graph
https://imgur.com/a/JQpLSIm

number of levels = 1 #The shortest path from root to the furthest vertex (root is vertex 0)
diameter = 4 #Longest path between any two vertices
"""


"""
Exercise 1b
"""





from collections import deque 
import numpy as np 
import timeit
from numba import jit, prange

Example_graph = np.load('graph_array_random.npy')

def bfsv_python(G, s): #s = source vertex, G = graph
    Q = deque()
    distance = [np.inf]*len(G) #Sets all distances to infin
    visited = [False]*len(G) #Set all vertices to unvisited


    distance[s] = 0 #Distance from source to source is 0
    visited[s] = True #Mark source vertex as visited
    Q.append(s)
    
    while Q: 
        u = Q.pop() #u = current vertex
        #distance.append(u)
        
        for w in G[u]:
            if not visited[w]:
                visited[w] = True
                distance[w] = distance[u] + 1 #Should be able to move out of for loop, and append to distance according to geeksforgeeks
                Q.append(w)
    return distance



@jit(nopython=True)
#Just removed unnecessary for loop and while loop condition. 
def bfsv_jit(G, s): #s = source vertex, G = graph
    Q = []
    distance = np.zeros(len(G), dtype=np.int32)-1
    visited = [False]*len(G) #Set all vertices to unvisited


    distance[s] = 0 #Distance from source to source is 0
    visited[s] = True #Mark source vertex as visited
    Q.append(s)
    
    while Q: 
        u = Q.pop() #u = current vertex
        #distance.append(u)
        
        for w in G[u]:
            if not visited[w]:
                visited[w] = True
                distance[w] = distance[u] + 1 #Should be able to move out of for loop, and append to distance according to geeksforgeeks
                Q.append(w)
    return distance


@jit(nopython=True, parallel=True)
def bfsv_parallel(G, s): #s = source vertex, G = graph
    Q = []
    distance = np.zeros(len(G), dtype=np.int32)-1
    visited = [False]*len(G) #Set all vertices to unvisited


    distance[s] = 0 #Distance from source to source is 0
    visited[s] = True #Mark source vertex as visited
    Q.append(s)
    
    while Q: 
        u = Q.pop() #u = current vertex
        #distance.append(u)
        
        for i in prange(len(G[u])):
            w = G[u][i]
            if not visited[w]:
                visited[w] = True
                
                distance[w] = distance[u] + 1 #Should be able to move out of for loop, and append to distance according to geeksforgeeks
                Q.append(w)
    return distance


#print(timeit.timeit(lambda: bfsv_jit(Example_graph, 0), number=3))
#print(timeit.timeit(lambda: bfsv_parallel(Example_graph, 0), number=3))

print(bfsv_jit(Example_graph, 0))
print(bfsv_parallel(Example_graph, 0))
#print(np.isclose(bfsv_parallel(Example_graph, 0), bfsv_python(Example_graph, 0)))