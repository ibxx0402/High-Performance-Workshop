import cv2
import numpy as np 
from numba import jit 
import timeit


""" 
image = np.array([[[1, 2, 3,], [4, 5, 6]],  #[2.185 5.185]
              [[0, 2, 3,], [4, 5, 6]]]) #[2.185 5.185] 
"""

def grayscale_np(image, grayscale_array):
    luminance_array = np.array([0.114, 0.587 , 0.299]) #in bgr format instead of rgb 
    np.matmul(image, luminance_array, out=grayscale_array)
    return grayscale_array

@jit(nopython=True)
def grayscale_jit(image, rows, grayscale_array):
    luminance_array = np.array([0.114, 0.587 , 0.299]) #in bgr format instead of rgb 

    for row in range(rows):
        grayscale_array[row] = np.sum(image[row]*luminance_array, axis=1)-128
        #print("\n", grayscale_array) """
    return grayscale_array


@jit(nopython=True)
def grayscale_jpeg_conv_jit(image, grayscale_array, rows, cols, k_width, k_height, quantization_matrix): 
    
    #Grayscale conversion
    luminance_array = np.array([0.114, 0.587 , 0.299]) #in bgr format instead of rgb 
    for row in range(rows):
        grayscale_array[row] = np.sum(image[row]*luminance_array, axis=1)-128
    
    #Partition the image into blocks of k_width x k_height
    for col_par in range(0, cols, k_width):
        for row_par in range(0, rows, k_height):
  
            block = grayscale_array[row_par:row_par+k_height, col_par:col_par+k_width]

            #First pass on rows 
            for row in range(k_height):
                y = np.empty(k_width)
                for k in range(k_width):
                    y_sum = 0
                    for n in range(k_width):
                        y_sum += block[row][n] * np.cos(((np.pi*k)*(2*n+1))/(2*(k_width)))
                    y[k] = 2 * y_sum
                block[row] = y

            #Second pass on cols 
            for col in range(k_width):
                y = np.empty(k_height)
                for k in range(k_height):
                    y_sum = 0
                    for n in range(k_height):
                        y_sum += block[:,col][n] * np.cos(((np.pi*k)*(2*n+1))/(2*(k_height)))
                    y[k] = 2 * y_sum
                block[:,col] = y

            #quantization
            block = np.round(block/quantization_matrix) * quantization_matrix

            #IDCT-II (DCT-III*1/N)
            #First pass on rows
            for row in range(k_height):
                y = np.empty(k_width)
                for k in range(k_width):
                    y_sum = (block[row][0])/2 
                    for n in range(1, k_width): #Important to start at 1
                        y_sum += block[row, n] * np.cos(((np.pi*n)*(2*k+1))/(2*(k_width)))
                    y[k] = (1/k_width) * y_sum
                block[row] = y
            #Second pass on cols 
            for col in range(k_width):
                y = np.empty(k_height)
                for k in range(k_height):
                    y_sum = (block[0,col])/2
                    for n in range(1, k_height):
                        y_sum += block[n, col] * np.cos(((np.pi*n)*(2*k+1))/(2*(k_height)))
                    y[k] =  (1/k_height) * y_sum
                block[:,col] = y
            grayscale_array[row_par:row_par+k_height, col_par:col_par+k_width] = block
    return grayscale_array


image = cv2.imread("Exercise_2/image.png")
rows, cols, colors = np.shape(image)
grayscale_array = np.empty((rows, cols))

k_width = 8
k_height = k_width

quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101], 
                                [72, 92, 95, 98, 112, 100, 103, 99]])


transformed_image = grayscale_jpeg_conv_jit(image, grayscale_array, rows, cols, k_width, k_height, quantization_matrix)


#image_np = grayscale_np(image)

#print("grayscale_jit ", timeit.timeit(lambda: grayscale_jit(image, rows, grayscale_array), number=10))
#print("grayscale_np ", timeit.timeit(lambda: grayscale_np(image, grayscale_array), number=10))


cv2.imwrite("compressed_image.png", transformed_image)
image_jit = grayscale_jit(image, rows, grayscale_array)
cv2.imwrite("grayscale_jit.png", image_jit)
#image = np.clip(transformed_image, 0, 255).astype(np.uint8)
#cv2.imshow("image", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()