import numpy as np

example_dag = np.array([[1, 2, 4], [3, 4, -1], [4, 5, -1], [6, -1, -1], [6, 7, 8], [7, -1, -1], [8, -1, -1], [8, -1, -1], [-1, -1, -1]])
np.save("graph_array_example.npy", example_dag)