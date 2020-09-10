#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:12:24 2020

@author: eddie
"""

import torch

class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, first_stack = False):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()
        
        
        self._w_ih = torch.nn.Linear(input_size, 4 * hidden_size)
        self._w_hh = torch.nn.Linear(hidden_size, 4 * hidden_size)
        self._activation = torch.nn.ReLU()
        
    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hx, cx = hidden[0], hidden[1]
        
        gates = self._w_ih(input) + self._w_hh(hx)
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 2)
        
        i_gate = torch.nn.functional.normalize(i_gate, p = 1, dim = 2)
        f_gate = torch.nn.functional.normalize(f_gate, p = 1, dim = 2)
        #c_gate = torch.nn.functional.normalize(c_gate, p = 1, dim = 2)
        o_gate = torch.nn.functional.normalize(o_gate, p = 1, dim = 2)
        
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        #c_gate = torch.tanh(c_gate)
        o_gate = torch.sigmoid(o_gate)
        ncx = (f_gate * cx) + (i_gate * c_gate)
        nhx = o_gate * self._activation(ncx)
        return nhx, ncx
    
class LSTM(torch.nn.Module):
    '''
    Input : 
        input_size : the size of the single time step length
        hidden_size : the size of encode feature size during LSTM
        nlayers : the number of stacked LSTM
    '''
    def __init__(self, input_size, hidden_size, nlayers = 1):
        super(LSTM, self).__init__()
        size = input_size
        cells = []
        for i in range(nlayers):
            _hidden_size = size
            cells.append(LSTMCell(_hidden_size, hidden_size))
            size = hidden_size
        self._lstm_cells = torch.nn.ModuleList(cells)
        self._nlayers = nlayers
        self._hidden_size = hidden_size
    
    '''
        Input :
            x : the input feature. Format should be (batch, time_len, feature_len)
            state : hidden state and cell state. Format should be (batch, layer_len, hidden_len)
    '''
    def forward(self, x, states):
        assert list(states[0].shape) == [x.shape[0], self._nlayers, self._hidden_size],\
            "Expect got hidden state size {} but got {}".format([x.shape[0], self._nlayers, self._hidden_size], list(states[0].shape))
        assert list(states[1].shape) == [x.shape[0], self._nlayers, self._hidden_size],\
            "Expect got cell state size {} but got {}".format([x.shape[0], self._nlayers, self._hidden_size], list(states[0].shape))
        
        pre_h_state = []
        for i in range(self._nlayers):
            nhx = []
            ncx = []
            for t in range(x.shape[1]):
                if t == 0:
                    hidden_state = states[0][:, i:i+1, :]
                    cell_state = states[1][:, i:i+1, :]
                else:
                    hidden_state = nhx[-1]
                    cell_state = ncx[-1]
                    
                if i == 0:
                    hidden_state, cell_state = self._lstm_cells[i](x[:, t:t+1], (hidden_state, cell_state))
                    pre_h_state.append(hidden_state)
                else:
                    hidden_state, cell_state = self._lstm_cells[i](pre_h_state[t], (hidden_state, cell_state))
                    pre_h_state[t] = hidden_state
                nhx.append(hidden_state)
                ncx.append(cell_state)
        
        return pre_h_state