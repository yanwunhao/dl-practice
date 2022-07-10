import torch
import torch.nn as nn

bs, T = 2, 3  # batch size, sequence length
input_size, hidden_size = 2, 3  # input feature size, hidden feature size
h_prev = torch.zeros(bs, hidden_size)  # hidden layer inits

input = torch.randn(bs, T, input_size)

rnn = nn.RNN(input_size, hidden_size, batch_first=True)
rnn_output, state_final = rnn(input, h_prev.unsqueeze(0))
print("pytorch rnn output:")
print(rnn_output, state_final)


# single direction RNN
def rnn_forward(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev):
    bs, T, input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs, T, h_dim)

    for t in range(T):
        x = input[:, t, :].unsqueeze(2)  # get features of time t bs*input_type*1
        w_ih_batch = weight_ih.unsqueeze(0).tile(bs, 1, 1)  # bs*h_dim*h_dim
        w_hh_batch = weight_hh.unsqueeze(0).tile(bs, 1, 1)  # bs*h_dim

        w_times_x = torch.bmm(w_ih_batch, x).squeeze(-1)  # bs*h_dim
        w_times_h = torch.bmm(w_hh_batch, h_prev.unsqueeze(2)).squeeze(-1)  # bs*h_dim
        h_prev = torch.tanh(w_times_x + bias_ih + w_times_h + bias_hh)

        h_out[:, t, :] = h_prev

    return h_out, h_prev.unsqueeze(0)


# for k,v in rnn.named_parameters():
#     print(k, v)

custom_rnn_output = rnn_forward(input, rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0, h_prev)
print("rnn_forward function output:")
print(custom_rnn_output)

bi_rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
h_prev = torch.zeros(2, bs, hidden_size)
bi_rnn_output, bi_rnn_state_final = bi_rnn(input, h_prev)
print("pytorch bi_rnn output:")
print(bi_rnn_output, bi_rnn_state_final)


# bidirectional RNN
def bidirectional_rnn_forward(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev, \
                              weight_ih_reverse, weight_hh_reverse, bias_ih_reverse, bias_hh_reverse, h_prev_reverse
                              ):
    bs, T, input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs, T, h_dim * 2)
    forward_output, _ = rnn_forward(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev)
    backward_output, _ = rnn_forward(torch.flip(input, [1]), weight_ih_reverse, weight_hh_reverse, bias_ih_reverse,
                                  bias_hh_reverse,
                                  h_prev_reverse)  # backward layer

    h_out[:, :, :h_dim] = forward_output
    h_out[:, :, h_dim:] = backward_output

    return h_out, h_out[:, -1, :].reshape((bs, 2, h_dim)).transpose(0, 1)


custom_bi_rnn_output = bidirectional_rnn_forward(input, bi_rnn.weight_ih_l0, bi_rnn.weight_hh_l0, bi_rnn.bias_ih_l0,
                                                 bi_rnn.bias_hh_l0, h_prev[0], bi_rnn.weight_ih_l0_reverse,
                                                 bi_rnn.weight_hh_l0_reverse, bi_rnn.bias_ih_l0_reverse,
                                                 bi_rnn.bias_hh_l0_reverse, h_prev[1])
print("bidirectional_rnn_forward function output")
print(custom_bi_rnn_output)
