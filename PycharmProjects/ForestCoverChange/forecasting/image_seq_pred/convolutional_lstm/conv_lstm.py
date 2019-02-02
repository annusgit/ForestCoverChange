

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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

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

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False,
                 bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

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

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
            # print(len(layer_output_list), len(last_state_list))
            return last_state_list[0], layer_output_list[0]

        return last_state_list, layer_output_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
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
    model = ConvLSTM(input_size=(height, width),
                     input_dim=channels,
                     hidden_dim=[10],
                     kernel_size=(3, 3),
                     num_layers=1,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False)
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
    height, width, channels, t_steps = 64, 64, 1, 3
    hidden_dimensions = [4, 8, 16]
    reverse_hidden = hidden_dimensions[::-1]
    num_layers = len(hidden_dimensions)
    forward_model = ConvLSTM(input_size=(height, width),
                             input_dim=channels,
                             hidden_dim=hidden_dimensions,
                             kernel_size=(3, 3),
                             num_layers=num_layers,
                             batch_first=True,
                             bias=True,
                             return_all_layers=False)

    reverse_model = ConvLSTM(input_size=(height, width),
                             input_dim=reverse_hidden[0],
                             hidden_dim=reverse_hidden,
                             kernel_size=(3, 3),
                             num_layers=num_layers,
                             batch_first=True,
                             bias=True,
                             return_all_layers=False)
    forward_model.eval()
    reverse_model.eval()
    # input order (b, t, c, h, w)
    forward_input = torch.Tensor(16, t_steps, channels, height, width)
    print('Forward Model:')
    [last_hidden, last_cell_state], last_output = forward_model(forward_input)

    print('\tforward_input.shape', forward_input.shape)
    print('\tlast_hidden.shape', last_hidden.shape)
    print('\tlast_cell_state.shape', last_cell_state.shape)
    print('\tlast_output.shape', last_output.shape)

    reverse_input = torch.Tensor(last_output)
    print('Reverse Model:')
    [last_hidden, last_cell_state], last_output = reverse_model(reverse_input,
                                                                input_hidden_state=[last_hidden,
                                                                                    last_cell_state])

    print('\treverse_input.shape', forward_input.shape)
    print('\tlast_hidden.shape', last_hidden.shape)
    print('\tlast_cell_state.shape', last_cell_state.shape)
    print('\tlast_output.shape', last_output.shape)
    pass


if __name__ == '__main__':
    # let's fix this thing
    # check_model_on_moving_mnist()
    check_model()













