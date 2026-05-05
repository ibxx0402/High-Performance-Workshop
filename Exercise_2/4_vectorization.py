import cv2
import numpy as np 
import timeit

def grayscale_np(image):
    luminance = np.array([0.114, 0.587, 0.299])
    rows, cols, colors = image.shape
    grayscale_array = np.empty((rows, cols))
    np.matmul(image, luminance, out=grayscale_array)
    return grayscale_array

def build_dct_matrix(K):
    n = np.arange(K, dtype=np.float32)
    k = n[:, None]
    D = np.cos(np.pi * k * (2 * n + 1) / (2 * K))
    # Normalize rows
    D[0, :] *= np.sqrt(1 / K)
    D[1:, :] *= np.sqrt(2 / K)
    return D

def jpeg_compress_global(image, quantization_matrix):

    # grayscale 
    luminance = np.array([0.114, 0.587, 0.299])
    grayscale_array = (image @ luminance).astype(np.float32) - 128.0

    rows, cols = grayscale_array.shape

    # build full DCT matrices for the image size
    Dh = build_dct_matrix(rows)
    Dw = build_dct_matrix(cols)

    # 2D DCT on entire image
    Y_dct = Dh @ grayscale_array @ Dw.T

    # Create tiled quantization matrix
    q_rows = (rows + 7) // 8  # number of 8x8 blocks in rows
    q_cols = (cols + 7) // 8
    quantization_matrix_tiled = np.tile(quantization_matrix, (q_rows, q_cols))
    quantization_matrix_tiled = quantization_matrix_tiled[:rows, :cols]  # crop to exact size

    # quantization
    Y_Q = np.round(Y_dct / quantization_matrix_tiled).astype(np.float32) * quantization_matrix_tiled

    # inverse DCT
    Y_rec = Dh.T @ Y_Q @ Dw
    rec = np.clip(Y_rec + 128.0, 0, 255).astype(np.uint8)

    return rec

def jpeg_compress_vectorized(image, k_width, k_height, quantization_matrix):

    # grayscale 
    luminance = np.array([0.114, 0.587, 0.299])
    grayscale_array = (image @ luminance).astype(np.float32) - 128.0

    rows, cols = grayscale_array.shape

    # crop to nearest multiple pf block size
    rows_crop = (rows // k_height) * k_height
    cols_crop = (cols // k_width) * k_width
    grascale_cropped = grayscale_array[:rows_crop, :cols_crop]

    # block partitioning    
    blocks = grascale_cropped.reshape(rows_crop // k_height, k_height, cols_crop // k_width, k_width).transpose(0, 2, 1, 3)
    
    # 2D DCT
    Dh = build_dct_matrix(k_height)
    Dw = build_dct_matrix(k_width)
    Y_dct = Dh @ blocks @ Dw.T

    # quantization
    Y_Q = np.round(Y_dct / quantization_matrix).astype(np.float32) * quantization_matrix

    # inverse DCT
    Y_rec = Dh.T @ Y_Q @ Dw
    rec = (Y_rec.transpose(0, 2, 1, 3).reshape(rows_crop, cols_crop))
    rec = np.clip(rec + 128.0, 0, 255).astype(np.uint8)

    return rec

# load the image 
image = cv2.imread('C:/Users/adc01/OneDrive - Aalborg Universitet/AAU/6semester/High Performance Programming/Workshop/High-Performance-Workshop/Exercise_2/grayscale.jpeg')

k_width, k_height = 8, 8

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
t1 = timeit.timeit(
    lambda: jpeg_compress_vectorized(image, k_height, k_width, quantization_matrix),
    number=10
)

t2 = timeit.timeit(
    lambda: jpeg_compress_global(image, quantization_matrix),
    number=10
)
print(f"Average time: {t1/10*1000:.2f} ms\n\nAverage time: {t2/10*1000:.2f} ms")

#image_np = grayscale_np(image)
#cv2.imwrite("grayscale.jpeg", image_np)


result = jpeg_compress_vectorized(image, k_height, k_width, quantization_matrix)
print(result.dtype)  # Should print: uint8
print(result.min(), result.max())
cv2.imwrite("compressed_image.jpeg", result)

# Global DCT version
result_global = jpeg_compress_global(image, quantization_matrix)
cv2.imwrite("compressed_image_global.jpeg", result_global)
