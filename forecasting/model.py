

from __future__ import print_function
from __future__ import division
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_dataloaders
from torchsummary import summary
from torchviz import make_dot


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hn = self.gru(x, self.initHidden(x.shape[0]))
        # hn has a shape => (num_layers * num_directions, batch, hidden_size) > (1, N, hidden_size)
        out = self.linear(hn.view(x.shape[0], -1))
        return out

    def initHidden(self, N):
        # hidden state size is (num_layers * num_directions, batch, hidden_size):
        return torch.randn(1, N, self.hidden_size)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    this_file = '/home/annus/Desktop/statistics_250m_16_days_NDVI.json'
    train, val, test = get_dataloaders(file_path=this_file, batch_size=2)
    gru = GRU(input_size=1, hidden_size=4, output_size=1)
    # the input is given as (batch_size, sequence_length, size_of_one_input_in_sequence)
    for data in train:
        x, y = data['input'].unsqueeze(2), data['label']
        output = gru(x)
    print(output.shape)
    pass


if __name__ == '__main__':
    main()





















