import torch.nn as nn
import torch
from torch.autograd import Variable
from .basic_blocks import SE_Block
from torch.nn import Module, Sequential, Conv2d


class CMSLSTM_cell(Module):
    def __init__(self, input_size, hidden_size, filter_size, img_size, ce_iterations=5):
        super(CMSLSTM_cell, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.hidden_size = hidden_size

        self.padding = int((filter_size - 1) / 2)  # in this way the output has the same size
        self._forget_bias = 1.0

        self.norm_cell = nn.LayerNorm([self.hidden_size, img_size, img_size])
        # Convolutional Layers
        self.conv_i2h = Sequential(
            Conv2d(self.input_size, 4 * self.hidden_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([4 * self.hidden_size, img_size, img_size])
        )
        self.conv_h2h = Sequential(
            Conv2d(self.hidden_size, 4 * self.hidden_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([4 * self.hidden_size, img_size, img_size])
        )

        # CE block
        self.ceiter = ce_iterations
        self.convQ = Sequential(
            Conv2d(self.hidden_size, self.input_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([self.input_size, img_size, img_size])
        )

        self.convR = Sequential(
            Conv2d(self.input_size, self.hidden_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([self.hidden_size, img_size, img_size])
        )

        # hidden states buffer, [h, c]
        self.hiddens = None

        # attention module
        self.SEBlock = SE_Block(self.hidden_size, img_size)
        self.attentions = None

        self.dropout = nn.Dropout(p=0.1)

    def CEBlock(self, xt, ht):
        for i in range(1, self.ceiter + 1):
            if i % 2 == 0:
                ht = (2 * torch.sigmoid(self.convR(xt))) * ht
            else:
                xt = (2 * torch.sigmoid(self.convQ(ht))) * xt

        return xt, ht

    def forward(self, x, init_hidden=False):
        # initialize the hidden states, consists of hidden state: h and cell state: c
        if init_hidden or (self.hiddens is None):
            self.init_hiddens(x)

        h, c = self.hiddens

        x, h = self.CEBlock(x, h)

        # caculate i2h, h2h
        i2h = self.conv_i2h(x)
        h2h = self.conv_h2h(h)

        (i, f, g, o) = torch.split(i2h + h2h, self.hidden_size, dim=1)

        # caculate next h and c
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + self._forget_bias)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        # ---------------------------------------------------
        next_c = f * c + i * g
        next_c = self.norm_cell(next_c)
        next_h = o * torch.tanh(next_c)

        next_h, next_c, self.attentions = self.SEBlock(next_h, next_c)

        self.hiddens = [next_h, next_c]

        return next_h

    def init_hiddens(self, x):
        b, c, h, w = x.size()

        self.hiddens = [Variable(torch.zeros(b, self.hidden_size, h, w)).cuda(),
                        Variable(torch.zeros(b, self.hidden_size, h, w)).cuda()]


class CMSLSTM(Module):
    def __init__(self, input_size, output_chans, hidden_size=128, filter_size=5, num_layers=4, img_size=64):
        super(CMSLSTM, self).__init__()
        self.n_layers = num_layers
        # embedding layer
        self.embed = Conv2d(input_size, hidden_size, 1, 1, 0)
        # lstm layers
        lstm = [CMSLSTM_cell(hidden_size, hidden_size, filter_size, img_size) for l in range(num_layers)]

        self.lstm = nn.ModuleList(lstm)
        # output layer
        self.output = Conv2d(hidden_size, output_chans, 1, 1, 0)

    def forward(self, x, init_hidden=False):
        h_in = self.embed(x)
        for l in range(self.n_layers):  # for every layer
            h_in = self.lstm[l](h_in, init_hidden)

        return self.output(h_in)


def get_cmslstm(input_chans=1, output_chans=1, hidden_size=64, filter_size=5, num_layers=4, img_size=64):
    model = CMSLSTM(input_chans, output_chans, hidden_size, filter_size, num_layers, img_size)
    return model
