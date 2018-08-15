

from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from torchviz import make_dot
import torch.nn.functional as F
from torchsummary import summary
from dataset import get_dataloaders

"""
    For the torch GRU, we have this:
        Constructor Parameters
            1. input_size:  The number of expected features in the input x
            2. hidden_size: The number of features in the hidden state h
    
        When calling a forward on a gru, remember this...
            1. input:  of shape (seq_len, batch, input_size) **
            2. h_0:    of shape (num_layers * num_directions, batch, hidden_size)
            3. output: of shape (seq_len, batch, num_directions * hidden_size) **
            4. h_n:    of shape (num_layers * num_directions, batch, hidden_size)
            ** (if batch_first = True, then input and output have batch size as the first dimension)
"""

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=batch_first)

    def forward(self, input, hidden):
        output = input
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        # hidden state size is (num_layers * num_directions, batch, hidden_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


    def initHiddenNormal(self, batch_size):
        # hidden state size is (num_layers * num_directions, batch, hidden_size):
        return torch.randn(self.num_layers, batch_size, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first=False):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=batch_first)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        output = self.out(torch.sum(output, dim=1))
        return output, hidden

    # def initHidden(self, batch_size):
    #     return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class ENC_DEC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_first=True):
        super(ENC_DEC, self).__init__()
        self.enc = EncoderRNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=batch_first)
        self.dec = DecoderRNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, output_size=output_size,
                              batch_first=batch_first)
        pass

    def forward(self, x):
        output, hn = self.enc(x, self.enc.initHiddenNormal(batch_size=x.shape[0]))
        output = torch.sum(output, dim=2).unsqueeze(2)
        output, hn = self.dec(output, hn)
        return output, hn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        print(hidden)
        out, hn = self.gru(x, hidden)
        # hn has a shape => (num_layers * num_directions, batch, hidden_size) > (1, N, hidden_size)
        # out = self.linear(hn.view(x.shape[0], -1))
        return out, hn

    def initHidden(self, N):
        # hidden state size is (num_layers * num_directions, batch, hidden_size):
        return torch.randn(1, N, self.hidden_size)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def main():
    in_seq_len, out_seq_len = 5, 5
    this_file = '/home/annus/Desktop/statistics_250m_16_days_NDVI.json'
    train, val, test = get_dataloaders(file_path=this_file, in_seq_len=in_seq_len,
                                       out_seq_len=out_seq_len, batch_size=8)

    enc_dec = ENC_DEC(input_size=1, hidden_size=4, num_layers=2, output_size=5, batch_first=True)
    for data in train:
        x, y = data['input'].unsqueeze(2), data['label']
        print(x.shape)
        output, hn = enc_dec(x)
        print(x.shape, output.shape, hn.shape)

    # encoder = EncoderRNN(input_size=1, hidden_size=4, num_layers=2, batch_first=True)
    # decoder = DecoderRNN(input_size=1, hidden_size=4, output_size=7, num_layers=2, batch_first=True)
    # the input is given as (batch_size, sequence_length, size_of_one_input_in_sequence)
    # for data in train:
    #     x, y = data['input'].unsqueeze(2), data['label']
    #     output, hn = encoder(x, encoder.initHidden(batch_size=x.shape[0]))
    #     output = torch.sum(output, dim=2).unsqueeze(2)
    #     output, hn = decoder(output, hn)
    #     print(x.shape, output.shape, hn.shape)
    pass


if __name__ == '__main__':
    main()





















