import torch
import torch.nn
import torch.nn.functional as F

import numpy as np

in_channel = 1
out_channel = 1

kernel_size = 3
bias = False

batch_szie = 1
input_size = [batch_szie, in_channel, 4, 4]

# implement in nn class
conv_layer = torch.nn.Conv2d(in_channel, out_channel, kernel_size, bias=bias)

input_feature_map = torch.randn(input_size)
output_feature_map_1 = conv_layer(input_feature_map)

print(f"weights: {conv_layer.weight}")
print(f"input: {input_feature_map}")
print(f"output_1: {output_feature_map_1}")

# implement in functional api
output_feature_map_2 = F.conv2d(input_feature_map, conv_layer.weight)
print(f"output_2: {output_feature_map_2}")


# implement convolutional operation with matrix operation
def matrix_multiplication_for_conv2d(input, kernel, bias=0, stride=1, padding=0):
    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))

    input_h, input_w = input.shape
    kernel_h, kernel_w = kernel.shape
    output_h = int(np.floor((input_h - kernel_h) / stride) + 1)
    output_w = int(np.floor((input_w - kernel_w) / stride) + 1)
    output = torch.zeros(output_h, output_w)

    for i in range(0, input_h - kernel_h + 1, stride):
        for j in range(0, input_w - kernel_w + 1, stride):
            computing_region = input[i:i+kernel_h, j:j+kernel_w]
            output[int(i/stride), int(j/stride)] = torch.sum(torch.mul(kernel, computing_region)) + bias

    return output


input_image = torch.randn(6, 5)
convolution_kernel = torch.randn(3, 3)
bias = torch.randn(1)

mat_mul_conv2d_output = matrix_multiplication_for_conv2d(input_image, convolution_kernel, bias=bias)
print(mat_mul_conv2d_output)

# compare to functional api conv2d
function_api_conv2d_output = F.conv2d(input_image.reshape((1, 1, input_image.shape[0], input_image.shape[1])),\
                                      convolution_kernel.reshape((1, 1, convolution_kernel.shape[0], convolution_kernel.shape[1])),\
                                      padding=0, bias=bias)
print(function_api_conv2d_output.squeeze(0).squeeze(0))
