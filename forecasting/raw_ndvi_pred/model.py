

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

##########################################################################################################
##########################                   Deprecated ;)                             ###################
##########################################################################################################
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
        # print(output.shape, hidden.shape)
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
        self.kill = nn.Dropout(p=0.8)
        self.non_lin = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=hidden_size)
        self.out1 = nn.Linear(hidden_size, hidden_size)
        self.out2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = F.relu(input)
        output, hidden = self.gru(output, hidden)
        # output = self.out(torch.sum(output, dim=1))
        output = self.non_lin(self.out1(output[:,0]))
        output = self.bn(output)
        output = self.kill(output)
        output = self.non_lin(self.out2(output))
        output = self.out(output)
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
        # output = torch.sum(output, dim=2).unsqueeze(2)
        output = output[:,:,0].unsqueeze(2)
        output, hn = self.dec(output, hn)
        return output, hn
##########################################################################################################


class EDGRU(nn.Module):
    '''
        Encoder-Decoder GRU model to use for learning the ndvi time sequence
    '''
    def __init__(self, input_size, hidden_size, num_layers, batch_first=False):
        super(EDGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # encoder and decoder GRUs
        self.egru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=batch_first)
        self.dgru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=batch_first)
        self.kill = nn.Dropout(p=0.8)
        self.non_lin = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=hidden_size)
        self.linear1 = nn.Linear(hidden_size*num_layers, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1) # outputs one real number as output

    def forward(self, x, hidden=None):
        if not isinstance(hidden, torch.Tensor):
            # set initial hidden state if it's None, else it must be coming from the previous time stamp!
            hidden = self.initHiddenNormal(x.shape[0]).to(torch.cuda.current_device())
        e_out, hn_e = self.egru(x, hidden) # encode
        # d_out, hn_d = self.dgru(torch.zeros_like(x), hn_e) # decode using encoder's final state
        # out = self.kill(out) # dpout
        pred = self.non_lin(self.linear1(hn_e.view(x.shape[0], -1))); hn_d = hn_e
        pred = self.bn(pred)
        pred = self.linear2(pred)
        return pred, hn_d

    def continuous_forward(self, x, out_seq_len):
        """
            Use this for continuous output sequence generation of arbitrarily long lengths
        :return: output of size seq_length, can generate arbitrarily long sequence of outputs
        :param x: only the input sequence to begin with
        """
        out_seq_gen = torch.Tensor().to(torch.cuda.current_device())
        # first output
        pred, hn = self.forward(x) # pred is the predicted next ndvi value, hn is the last hidden state
        out_seq_gen = torch.cat((out_seq_gen, pred), dim=1)
        for i in range(out_seq_len-1):
            x = pred.unsqueeze(2) #
            # x = torch.cat((x.squeeze(2)[:, 1:], pred), dim=1).unsqueeze(2)
            out_seq_gen = torch.cat((out_seq_gen, pred), dim=1)
            pred, hn = self.forward(x, hn)
        return out_seq_gen, hn

    def initHiddenNormal(self, N):
        # hidden state size is (num_layers * num_directions, batch, hidden_size):
        return torch.randn(self.num_layers, N, self.hidden_size)

    def initHiddenZero(self, batch_size):
        # hidden state size is (num_layers * num_directions, batch, hidden_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def lstm_test():
    class LST_model(nn.Module):

        def __init__(self):
            super(LST_model, self).__init__()
            self.lstm = nn.LSTM(input_size=3, hidden_size=3)
            pass

        def forward(self, x):

            pass

    lstm = LST_model()
    lstm.eval()
    inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5
    # initialize the hidden state.
    hidden = (torch.randn(1, 1, 3),
              torch.randn(1, 1, 3))


@torch.no_grad()
def main():
    in_seq_len, out_seq_len = 5, 5
    this_file = "statistics_250m_16_days_NDVI.json"
    train, val, test = get_dataloaders(file_path=this_file, in_seq_len=in_seq_len,
                                       out_seq_len=out_seq_len, batch_size=128)
    #
    # enc_dec = ENC_DEC(input_size=1, hidden_size=4, num_layers=2, output_size=5, batch_first=True)
    # gru = GRU(input_size=1, hidden_size=4, num_layers=2, output_size=5, batch_first=True)
    # for data in train:
    #     x, y = data['input'].unsqueeze(2), data['label']
    #     output, hn = gru(x)
    #     print(x.shape, output.shape, hn.shape)

    encoder = EncoderRNN(input_size=1, hidden_size=4, num_layers=1, batch_first=True)
    decoder = DecoderRNN(input_size=1, hidden_size=4, output_size=7, num_layers=1, batch_first=True)
    gru = EDGRU(input_size=1, hidden_size=4, num_layers=1, batch_first=True)
    gru.eval()

    # test = torch.Tensor(4, 8, 1)
    # predictions = gru.continous_forward(x=test, out_seq_len=3)
    # # the input is given as (batch_size, sequence_length, size_of_one_input_in_sequence)
    for data in train:
        x, y = data['input'].unsqueeze(2), data['label']
        output, hn = gru.continuous_forward(x=x, out_seq_len=15)
        # output = torch.sum(output, dim=2).unsqueeze(2)
        # output, hn = decoder(output, hn)
        print('log: input shape = {}, output shape = {}, hidden shape = {}'.format(x.shape, output.shape, hn.shape))
    # print(output[:,-1,:], hn)

    pass


if __name__ == '__main__':
    main()





















