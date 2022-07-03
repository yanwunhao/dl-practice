import torch
import numpy as np

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

print(x_data)
print(type(data), type(x_data), x_data.dtype)

data_np = np.random.normal(2)
x_data_np = torch.tensor(data_np)

print(x_data_np)
print(type(data_np), type(x_data_np), x_data_np.dtype)

x_ones = torch.ones_like(x_data)
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_ones, '\n', x_rand)

tensor_of_certain_shape = torch.rand([3, 3])
print(tensor_of_certain_shape)
print(tensor_of_certain_shape.dtype, tensor_of_certain_shape.shape)

# move our tensor to the GPU if available
tensor = torch.ones((4, 4))
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
print(tensor)

print(torch.is_tensor(tensor))
print(torch.is_complex(tensor))
print(torch.is_floating_point(tensor))
print(torch.numel(tensor))

zeros = torch.zeros((6, 6))
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1., 2, 3]).dtype)

print(torch.arange(0, 5, 2))

print(torch.eye(5, 3))

print(torch.full((2, 2), 6))

x = torch.randn(2, 3)
print(torch.cat([x, x], dim=0))
print(torch.cat([x, x], dim=1))
