

# an implementation of the convolutional lstm in pytorch taken from
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim  # number of channels in the input tensor
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim+self.hidden_dim, out_channels=4*self.hidden_dim,
                              kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, encoder_hidden_dim, decoder_hidden_dim, kernel_size, num_layers,
                 in_seq_len, out_seq_len, out_classes=2, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        encoder_hidden_dim = self._extend_for_multilayer(encoder_hidden_dim, num_layers)
        decoder_hidden_dim = self._extend_for_multilayer(decoder_hidden_dim, num_layers)
        if not len(kernel_size) == len(encoder_hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        if not len(kernel_size) == len(decoder_hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.out_classes = out_classes
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        encoder_cell_list = []
        decoder_cell_list = []
        self.classification_conv_1 = nn.Conv2d(in_channels=(self.out_seq_len*self.decoder_hidden_dim[0]),
                                               out_channels=(self.out_seq_len*self.decoder_hidden_dim[0]),
                                               kernel_size=3, padding=1)
        # encoder cell list
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.encoder_hidden_dim[i-1]
            encoder_cell_list.append(ConvLSTMCell(input_size=(self.height, self.width), input_dim=cur_input_dim,
                                                  hidden_dim=self.encoder_hidden_dim[i], kernel_size=self.kernel_size[i],
                                                  bias=self.bias))
        # decoder cell list
        for i in range(0, self.num_layers):
            cur_input_dim = self.decoder_hidden_dim[-i]
            # cur_input_dim = self.hidden_dim[-1] if i == 0 else self.hidden_dim[-1-i]
            decoder_cell_list.append(ConvLSTMCell(input_size=(self.height, self.width), input_dim=cur_input_dim,
                                                  hidden_dim=self.decoder_hidden_dim[i-1], kernel_size=self.kernel_size[i],
                                                  bias=self.bias))
        self.encoder_cell_list = nn.ModuleList(encoder_cell_list)
        self.decoder_cell_list = nn.ModuleList(decoder_cell_list)

    def forward(self, input_tensor, input_hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        # Implement stateful ConvLSTM
        if input_hidden_state is not None:
            # raise NotImplementedError()
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
            hidden_state[0] = input_hidden_state  # this is a tuple of [hidden, cell]
            print('Initializing with a given hidden state...')
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        # encoder forward pass
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.encoder_cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
        encoder_output = layer_output
        encoder_last_state = [h, c]
        # decoder forward pass; uses the last [hidden, cell] tuple
        cur_layer_input = torch.zeros(encoder_output.size(0), 1, encoder_output.size(2), encoder_output.size(3),
                                      encoder_output.size(4))
        hidden_state = self._init_hidden(batch_size=encoder_output.size(0))  # the last output is the input here
        hidden_state[0] = encoder_last_state  # this is a tuple of [hidden, cell], the last state is initial state
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(self.out_seq_len):
                h, c = self.decoder_cell_list[layer_idx](input_tensor=cur_layer_input[:,0,:,:,:], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            shape = layer_output.shape
            print(shape)
            batch, time_steps, channels, width, height = shape
            cur_layer_input = layer_output
            # take output predictions at the last layer of the decoder
            branch_output = self.classification_conv_1(layer_output.view(batch, time_steps*channels,
                                                                         width, height))
        decoder_output = layer_output
        decoder_last_state = [h, c]
        return encoder_last_state, encoder_output, decoder_last_state, decoder_output

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.encoder_cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def get_shapes(structure):
    for struct in structure:
        if isinstance(struct, torch.Tensor):
            print(struct.shape)
            return
        else:
            print(type(struct))
        get_shapes(struct)


@torch.no_grad()
def check_model_on_moving_mnist():
    height, width, channels = 64, 64, 1
    model = ConvLSTM(input_size=(height, width), input_dim=channels, hidden_dim=[8, 8], kernel_size=(3, 3),
                     num_layers=1, batch_first=True, bias=True, return_all_layers=False)
    model.eval()
    # input (b, t, c, h, w)
    moving_mnist = np.load('../moving_mnist/data.npy')
    moving_mnist = np.expand_dims(moving_mnist, axis=4).transpose(0,1,4,3,2)
    moving_mnist_tensor = torch.Tensor(moving_mnist)
    test_input = moving_mnist_tensor[0:16, ]
    print('test_input.shape = ', test_input.shape)
    hidden_list, output_list = model(test_input)
    print('len(output_list), len(hidden_list) = ', len(output_list), len(hidden_list))
    pass


@torch.no_grad()
def check_model():
    height, width, channels, num_layers, input_t_steps, out_t_steps = 64, 64, 1, 2, 4, 3
    encoder_hidden_dimensions = [8]*num_layers
    decoder_hidden_dimensions = [8]*num_layers
    reverse_hidden = encoder_hidden_dimensions[::-1]
    num_layers = len(encoder_hidden_dimensions)
    forward_lstm = ConvLSTM(input_size=(height, width), input_dim=channels, in_seq_len=input_t_steps,
                            out_seq_len=out_t_steps, kernel_size=(3, 3), num_layers=num_layers, batch_first=True,
                            encoder_hidden_dim=encoder_hidden_dimensions, decoder_hidden_dim=decoder_hidden_dimensions,
                            bias=True, return_all_layers=False)
    # reverse_lstm_encoder = ConvLSTM(input_size=(height, width), input_dim=reverse_hidden[0], hidden_dim=reverse_hidden,
    #                                 kernel_size=(3, 3), num_layers=num_layers, batch_first=True, bias=True,
    #                                 return_all_layers=False)
    forward_lstm.eval()
    # reverse_lstm_encoder.eval()
    # input order (b, t, c, h, w)
    forward_input = torch.Tensor(16, input_t_steps, channels, height, width)
    print('Forward Model:')
    encoder_last_states, encoder_last_output, decoder_last_states, decoder_last_output = forward_lstm(forward_input)
    [encoder_last_hidden, encoder_last_cell_state] = encoder_last_states
    [decoder_last_hidden, decoder_last_cell_state] = encoder_last_states
    print('\tforward_input.shape', forward_input.shape)
    print('\tencoder_last_hidden.shape', encoder_last_hidden.shape)
    print('\tencoder_last_cell_state.shape', encoder_last_cell_state.shape)
    print('\tencoder_last_output.shape', encoder_last_output.shape)
    print('\tdecoder_last_hidden.shape', decoder_last_hidden.shape)
    print('\tdecoder_last_cell_state.shape', decoder_last_cell_state.shape)
    print('\tdecoder_last_output.shape', decoder_last_output.shape)
    # reverse_input = torch.Tensor(last_output)
    # print('Reverse Model:')
    # [last_hidden, last_cell_state], last_output = reverse_lstm_encoder(reverse_input,
    #                                                                    input_hidden_state=[last_hidden,
    #                                                                                        last_cell_state])
    # print('\treverse_input.shape', reverse_input.shape)
    # print('\tlast_hidden.shape', last_hidden.shape)
    # print('\tlast_cell_state.shape', last_cell_state.shape)
    # print('\tlast_output.shape', last_output.shape)
    pass


if __name__ == '__main__':
    # let's fix this thing
    # check_model_on_moving_mnist()
    check_model()













