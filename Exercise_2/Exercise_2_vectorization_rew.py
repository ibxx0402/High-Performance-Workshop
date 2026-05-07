import cv2
import numpy as np 
import timeit
from numba import jit 

def jpeg_compress_vectorized(image, grayscale_array, rows, cols, k_width, k_height, quantization_matrix):

    # grayscale 
    luminance = np.array([0.114, 0.587, 0.299], dtype=np.float32) #BGR format 
    np.matmul(image, luminance, out=grayscale_array)
    grayscale_array -= 128

    # block partitioning    
    blocks = grayscale_array.reshape(rows // k_height, k_height, cols // k_width, k_width).transpose(0, 2, 1, 3)

    # DCT-II matrices
    n_h = np.arange((k_height), dtype=np.float32)
    k_h = np.arange((k_height), dtype=np.float32)
    dct_cos_h = 2 * np.cos(np.pi * k_h[:, None] * (2 * n_h[None, :] + 1) / (2 * k_height))

    n_w = np.arange((k_width), dtype=np.float32)
    k_w = np.arange((k_width), dtype=np.float32)
    dct_cos_w = 2 * np.cos(np.pi * k_w[:, None] * (2 * n_w[None, :] + 1) / (2 * k_width))

    # DCT
    Y_dct = dct_cos_h @ blocks @ dct_cos_w.T

    # Quantization
    Y_dct = np.round(Y_dct / quantization_matrix)* quantization_matrix


    # IDCT-II matrices
    idct_cos_h = np.cos(np.pi * n_h[None, :] * (2 * k_h[:, None] + 1) / (2 * k_height)) / k_height
    idct_cos_h[:, 0] /= 2 

    idct_cos_w = np.cos(np.pi * n_w[None, :] * (2 * k_w[:, None] + 1) / (2 * k_width)) / k_width
    idct_cos_w[:, 0] /= 2
    
    # IDCT
    idct = idct_cos_h @ Y_dct @ idct_cos_w.T

    image = (idct.transpose(0, 2, 1, 3).reshape(rows, cols))

    return image + 128

# load the image 
image = cv2.imread('Exercise_2/image.png')

rows, cols, colors = np.shape(image)
grayscale_array = np.empty((rows, cols))

k_width = 8
k_height = k_width

# Quantization matrix (JPEG luminance)
quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.float32)

# timing

#"""
number = 20
print("jpeg_compress_vectorized ", timeit.timeit(lambda: jpeg_compress_vectorized(image, grayscale_array, rows, cols, k_width, k_height, quantization_matrix), number=number)/number)
exit() 
#"""

transformed_image = jpeg_compress_vectorized(image, grayscale_array, rows, cols, k_width, k_height, quantization_matrix)
cv2.imwrite("compressed_image3.png", transformed_image)

cv2.imshow("image", transformed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
