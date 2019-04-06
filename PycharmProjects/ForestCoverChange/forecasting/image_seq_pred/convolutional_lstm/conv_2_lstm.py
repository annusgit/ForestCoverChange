

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = int((kernel_size - 1) / 2)
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding,  bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]))
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], bias=True):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.bias = bias
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.bias)
            # setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[:,step,:,:,:]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                # name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    # (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                    #                                          shape=(height, width))
                    (h, c) = self._all_layers[i].init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))
                # do forward
                (h, c) = internal_state[i]
                # this is calling the ConvLSTMCell
                # x, new_c = getattr(self, name)(x, h, c)
                x, new_c = self._all_layers[i](x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)
        return outputs, (x, new_c)


if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=1, hidden_channels=[10, 8, 2], kernel_size=3, step=4,
                        effective_step=[0,1])  # effective step 1 means the last step I want to record
    loss_fn = torch.nn.MSELoss()
    # the input size is [batch, time_steps, channels, height, width]
    input = Variable(torch.randn(1, 4, 1, 64, 64))
    target = Variable(torch.randn(1, 2, 64, 64)).double()
    outputs, [hidden, cell] = convlstm(input)
    output_1 = outputs[0].double()
    output_2 = outputs[0].double()
    print((torch.eq(output_1, output_2).sum()/output_2.numel()).item())
    res = torch.autograd.gradcheck(loss_fn, (output_1, target), eps=1e-6, raise_exception=True)
    print(res)














