import torch
import torch.nn
import torch.nn.functional as F

import numpy as np


def matrix_multiplication_for_conv2d_flatten(input, kernel, bias=0, stride=1, padding=0):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    input_h, input_w = input.shape
    kernel_h, kernel_w = kernel.shape
    output_h = int(np.floor((input_h - kernel_h) / stride) + 1)
    output_w = int(np.floor((input_w - kernel_w) / stride) + 1)
    output = torch.zeros(output_h, output_w)

    region_matrix = torch.zeros(output.numel(), kernel.numel())
    kernel_matrix = kernel.reshape((kernel.numel(), 1))  # column vector representation of kernel

    row_index = 0
    for i in range(0, input_h - kernel_h + 1, stride):
        for j in range(0, input_w - kernel_w + 1, stride):
            computing_region = input[i:i + kernel_h, j:j + kernel_w]
            region_vector = torch.flatten(computing_region)
            region_matrix[row_index] = region_vector
            row_index = row_index + 1

    output_matrix = torch.mm(region_matrix, kernel_matrix)
    output = output_matrix.reshape((output_h, output_w)) + bias

    return output


input_image = torch.randn(6, 6)
convolution_kernel = torch.randn(3, 3)
bias = torch.randn(1)

mat_mul_conv2d_output = matrix_multiplication_for_conv2d_flatten(input_image, convolution_kernel, bias=bias)
print(mat_mul_conv2d_output)

function_api_conv2d_output = F.conv2d(input_image.reshape((1, 1, input_image.shape[0], input_image.shape[1])), \
                                      convolution_kernel.reshape(
                                          (1, 1, convolution_kernel.shape[0], convolution_kernel.shape[1])), \
                                      bias=bias
                                      )
print(function_api_conv2d_output.squeeze(0).squeeze(0))
