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
from numba import jit, prange, config
import os



cpuCount = os.cpu_count()

#config.NUMBA_NUM_THREADS = 4
#cpuCount = 4

print("Number of CPUs in the system:", cpuCount)

def bfsv_python(G, s): #s = source vertex, G = graph
    V = len(G)
    distance = np.full(V, -1, dtype=np.int32)
    distance[s] = 0
    frontier = 1

    frontier_array = np.zeros(V, dtype=np.bool_)
    frontier_array[s] = True

    while np.any(frontier_array):
        neighbor_array = np.zeros(V, dtype=np.bool_)

        for u in range(V): #Can be length V instead as every FoundS is V length         
            if frontier_array[u]: #Find actual node
                for v in G[u]:
                    if distance[v] == -1:
                        neighbor_array[v] = True      
                        distance[v] = frontier

        frontier_array = neighbor_array
        frontier += 1
    return distance

@jit(nopython=True, parallel=True)
def bfsv_parallel(graph, source):
    V = len(graph)
    distance = np.full(V, -1, dtype=np.int32)
    distance[source] = 0
    frontier = 1

    frontier_array = np.zeros(V, dtype=np.bool_)
    frontier_array[source] = True 

    while np.any(frontier_array):
        neighbor_array = np.zeros(V, dtype=np.bool_) # The Visited nodes in this itteration

        for u in prange(V): #Can be length V instead as every FoundS is V length         
            if frontier_array[u]: #Find actual node
                for v in graph[u]:
                    if distance[v] == -1:
                        neighbor_array[v] = True      
                        distance[v] = frontier

        frontier_array = neighbor_array # The now visited notes
        frontier += 1

    return distance

@jit(nopython=True)
#Just removed unnecessary for loop and while loop condition. 
def bfsv_jit(G, s): #s = source vertex, G = graph
    V = len(G)
    distance = np.full(V, -1, dtype=np.int32)
    distance[s] = 0
    frontier = 1

    frontier_array = np.zeros(V, dtype=np.bool_)
    frontier_array[s] = True

    while np.any(frontier_array):
        neighbor_array = np.zeros(V, dtype=np.bool_)

        for u in range(V): #Can be length V instead as every FoundS is V length         
            if frontier_array[u]: #Find actual node
                for v in G[u]:
                    if distance[v] == -1:
                        neighbor_array[v] = True      
                        distance[v] = frontier

        frontier_array = neighbor_array
        frontier += 1

    return distance


if __name__ == "__main__":

    
    Example_graph = np.load('graph_array_random_g2.npy')
    
    test_array = [bfsv_jit, bfsv_python, bfsv_parallel]
    test_names = ["bfsv_jit", "bfsv_python", "bfsv_parallel"]
    
    for index, test in enumerate(test_array):
        test_array[index] = timeit.timeit(lambda: test(Example_graph, 0), number=3)
        print(f"{test_names[index]} {test_array[index]}")
    
    #Pick fastest sequential version
    if float(test_array[0]) <= float(test_array[1]):
        t1 = test_array[0]
    else:
        t1 = test_array[1]
    
    speedup = t1/float(test_array[2])
    print(f"speedup = {speedup}")
    print(f"effeciency = {speedup/cpuCount}")
    
    #Tests with m1 pro, with 4 performance and 4 effeciency cores 


"""
For graph with 25_000 nodes and 24_999 neighbors from graph_creation.py
bfsv_jit 5.434447417006595
bfsv_python 201.06373858402367
bfsv_parallel 0.9579067500017118
speedup = 5.673253077083843
effeciency = 0.7091566346354804
"""

"""
For graph with 100_000 nodes and 3 neighbors from graph_creation.py
Number of CPUs in the system: 8
bfsv_jit 0.32240383300813846
bfsv_python 0.4663189999992028
bfsv_parallel 0.3625497089815326
- speedup = 0.8892679404262352
- effeciency = 0.1111584925532794
bfsv_dist_mem_parallel = 8.732920917012962
- speedup = 0.05951265767
- effeciency = 0.007439082209
"""

"""
For graph with 1_000_000 nodes and 3 neighbors from graph_creation.py
Number of CPUs in the system: 8
bfsv_jit 0.5197193329804577
bfsv_python 5.412401041015983
bfsv_parallel 0.4219777919934131
- speedup = 1.2316272155587047
- effeciency = 0.1539534019448381
bfsv_dist_mem_parallel = 1033.3555122080143
- speedup = 0.0005029433983
- effeciency = 0.00006286792479
"""

"""
For example graph
Number of CPUs in the system: 8
bfsv_jit 0.38280683298944496
bfsv_python 0.00011183301103301346
bfsv_parallel 0.38762529200175777
- speedup = 0.00028850803428096824
- effeciency = 3.606350428512103e-05
bfsv_dist_mem_parallel = 0.14827950001927093 p = 3 unable to run higher num
- speedup = 0.0007542041281
- effeciency = 0.00009427551602
"""