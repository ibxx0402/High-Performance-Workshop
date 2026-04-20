#Creation of a adjecency matrix for a grapth
# Created from a list of, containing a lists of edges between vertices, each index of the outer list represents a vertex. The inner list contains the index of the vertices.


vertex0 = [1, 2, 3]
vertex1 = [0]
vertex2 = [0, 3]
vertex3 = [0, 2]
Example_graph = [vertex0, vertex1, vertex2, vertex3] 

"""
link to graph
https://imgur.com/a/JQpLSIm

number of levels = 1 #The shortest path from root to the furthest vertex (root is vertex 0)
diameter = 4 #Longest path between any two vertices
"""



"""
Exercise 1a
Explain why the total work done by your BFS procedure is Θ(𝑉 + 𝐸). Discuss the
“depth” of BFS in a parallel sense, focusing on the number of BFS levels (which
often equals the graph’s diameter or a bound on it)

The reason the Big Theta of bfs on a graph is Θ(𝑉 + 𝐸) is because the algorithm needs to at minimum reach all of the vertices and edges 
at least once to ensure the optimal solution has been found. 

Not sure if understood correctly, but the depth of the depth of a parallel BFS can be 1 as long as there are enough processors as each vertex can be processed independently. 

"""


"""
Exercise 1b
"""
from collections import deque 
import numpy as np 


def bfs(G, s): #s = source vertex, G = graph
    Q = deque()
    distance = [0]*len(G)
    visited = [0]*len(G)

    for vertex in range(len(G)):
        distance[vertex] = np.inf
        visited[vertex] = False

    distance[s] = 0
    visited[s] = True
    Q.append(s)

    while len(Q) != 0:
        u = Q.pop()
        for w in G[u]:
            if not visited[w]:
                visited[w] = True
                distance[w] = distance[u] + 1
                Q.append(w)
    return distance



#Just removed unnecessary for loop and while loop condition. 
def bfsv2(G, s): #s = source vertex, G = graph
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

print(bfsv2(Example_graph, 0))