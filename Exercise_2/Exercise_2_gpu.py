import torch
import numpy as np
import cv2
import timeit

if torch.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("No hardware acc")
    exit()

print(f'Using {device}')

def grayscale_jpeg_conv_torch(image, grayscale_tensor, quantization_matrix, k_width, k_height, rows, cols, device):
    luminance_tensor = torch.tensor([0.114, 0.587, 0.299], device=device)
    torch.matmul(image, luminance_tensor, out=grayscale_tensor)
    grayscale_tensor -= 128
    

    blocks = torch.permute(torch.reshape(grayscale_tensor, (rows // k_height, k_height, cols // k_width, k_width)), (0, 2, 1, 3))
    
    # DCT-II matrices
    n_h = torch.arange(k_height, device=device, dtype=torch.float32)
    k_h = torch.arange(k_height, device=device, dtype=torch.float32)
    dct_cos_h = 2 * torch.cos(torch.pi * k_h[:, None] * (2 * n_h[None, :] + 1) / (2 * k_height))

    n_w = torch.arange(k_width, device=device, dtype=torch.float32)
    k_w = torch.arange(k_width, device=device, dtype=torch.float32)
    dct_cos_w = 2 * torch.cos(torch.pi * k_w[:, None] * (2 * n_w[None, :] + 1) / (2 * k_width))

    # DCT
    Y_dct = dct_cos_h @ blocks @ dct_cos_w.T

    # Quantization
    Y_dct = torch.round(Y_dct / quantization_matrix) * quantization_matrix

    # IDCT-II matrices
    idct_cos_h = torch.cos(torch.pi * n_h[None, :] * (2 * k_h[:, None] + 1) / (2 * k_height)) / k_height
    idct_cos_h[:, 0] /= 2 

    idct_cos_w = torch.cos(torch.pi * n_w[None, :] * (2 * k_w[:, None] + 1) / (2 * k_width)) / k_width
    idct_cos_w[:, 0] /= 2

    # IDCT
    idct = idct_cos_h @ Y_dct @ idct_cos_w.T
    image = idct.permute(0, 2, 1, 3).reshape(rows, cols)
    return image+128

    

image = cv2.imread("Exercise_2/image.png")

image = torch.from_numpy(image.astype(np.float32)).to(device)
rows, cols, colors = image.shape
grayscale_tensor = torch.zeros((rows, cols), device=device) 

k_width = 8
k_height = k_width

quantization_matrix = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]], device=device)



#"""
number = 20
print("grayscale_jpeg_conv_torch ", timeit.timeit(lambda: grayscale_jpeg_conv_torch(image, grayscale_tensor, quantization_matrix, k_width, k_height, rows, cols, device), number=number)/number)
exit() 
#"""

transformed_image = grayscale_jpeg_conv_torch(image, grayscale_tensor, quantization_matrix, k_width, k_height, rows, cols, device)
transformed_image = transformed_image.cpu().numpy()



cv2.imshow("image", transformed_image.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()