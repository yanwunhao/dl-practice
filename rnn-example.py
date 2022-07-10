import torch
import torch.nn as nn


# single direction and single layer rnn
single_layer_rnn = nn.RNN(4, 3, 1, batch_first=True)
input = torch.randn(1, 2, 4)
output, h_n = single_layer_rnn(input)
print(input)
print("single layer RNN")
print(output)

# bidirectional single layer rnn
bidirectional_rnn = nn.RNN(4, 3, 1, batch_first=True, bidirectional=True)
output, h_n = bidirectional_rnn(input)
print("bidirectional single layer RNN")
print(output)