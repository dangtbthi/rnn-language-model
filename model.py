import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import collections
import random

from tqdm import tqdm

class MyRNN(nn.Module):
    def __init__(self, num_inputs, num_hiddens, vocab_size):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.vocab_size = vocab_size

        self.hidden = nn.Linear(self.num_inputs + self.num_hiddens, self.num_hiddens)
        self.output = nn.Linear(self.num_hiddens, self.vocab_size)

    def forward(self, inputs, state = None):
        '''
        inputs: (time_step, batch_size, num_nputs)
        state: (batch_size, num_hiddens)
        '''
        if state is None:
            state = torch.zeros((inputs.shape[1], self.num_hiddens))
        outputs = []
        for X in inputs:
            embs = F.one_hot(X.T, num_classes = self.vocab_size).type(torch.float32)
            combined = torch.cat((embs, state), dim = 1)
            state = torch.tanh(self.hidden(combined))
            output = self.output(state)
            outputs.append(output)
        return outputs, state

loss_fn = nn.CrossEntropyLoss()
