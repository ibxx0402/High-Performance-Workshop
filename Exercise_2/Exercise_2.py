import cv2
import numpy as np 
from numba import jit 
import timeit


""" 
image = np.array([[[1, 2, 3,], [4, 5, 6]],  #[2.185 5.185]
              [[0, 2, 3,], [4, 5, 6]]]) #[2.185 5.185] 
"""

@jit(nopython=True)
def grayscale_jit(image, rows, grayscale_array):
    luminance_array = np.array([0.114, 0.587 , 0.299]) #in bgr format instead of rgb 

    for row in range(rows):
        grayscale_array[row] = np.sum(image[row]*luminance_array, axis=1)-128
        #print("\n", grayscale_array) """
    return grayscale_array


def grayscale_np(image, grayscale_array):
    luminance_array = np.array([0.114, 0.587 , 0.299]) #in bgr format instead of rgb 
    np.matmul(image, luminance_array, out=grayscale_array)
    return grayscale_array

#@jit(nopython=True)
def partitioning(image, rows, cols, k_width , k_height): 
    for col_par in range(0, cols, k_width):
        for row_par in range(0, rows, k_height):
  
            block = image[row_par:row_par+k_height, col_par:col_par+k_height]

            for row in range(k_width):
                y = np.empty(k_width)
                for k in range(k_width):
                    y_sum = 0
                    for n in range(k_width):
                        y_sum += block[row][n] * np.cos(((np.pi*k)*(2*n+1))/(2*(k_width-1)))
                    y[k] = 2 * y_sum
                block[row] = y
            #image[row_par, col_par] = block
            #print(block)

    return 


image = cv2.imread("Exercise_2/image.png")
rows, cols, colors = np.shape(image)
grayscale_array = np.empty((rows, cols))

k_width = 8
k_height = k_width


image_jit = grayscale_jit(image, rows, grayscale_array)

partition_count = (rows*cols)//(k_width*k_height)
partitioning(image_jit, rows, cols, k_width, k_height)

#rows_res, cols_res = np.shape(image_jit)
#print((rows_res*cols_res)//(k_width*k_height))

#image_np = grayscale_np(image)

#print("grayscale_jit ", timeit.timeit(lambda: grayscale_jit(image, rows, grayscale_array), number=10))
#print("grayscale_np ", timeit.timeit(lambda: grayscale_np(image, grayscale_array), number=10))


""" cv2.imshow("image", image)

cv2.waitKey(0)

cv2.destroyAllWindows() """