from math import sqrt
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import embedding


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNLayer, self).__init__()

        self.hidden_size = hidden_size

        # TODO: Initialize weights and biases
        # Initialize weights with Xavier initialization
        # For weight matrix of shape (n_in, n_out):
        # Normal distribution with mean 0 and variance 2 / (n_in + n_out)
        # Use torch.randn
        # Initialize biases with zeros (torch.zeros)
        # Use nn.Parameter to create the weights and biases and pass in the initialziation
        self.W_xh = torch.randn((input_size, hidden_size)) * sqrt(
            2 / (input_size + hidden_size)
        )
        self.W_hh = torch.randn((hidden_size, hidden_size)) * sqrt(1 / hidden_size)
        self.b_h = torch.zeros(hidden_size)
        self.W_hy = torch.randn((hidden_size, output_size)) * sqrt(
            2 / (hidden_size + output_size)
        )
        self.b_y = torch.zeros(output_size)

        # TODO: Initialize activation function
        # Use nn.Tanh
        self.tanh = nn.Tanh()

    # TODO: Implement forward pass
    def forward(self, x):
        # Define dimensions
        B = x.shape[0]
        T = x.shape[1]
        H = x.shape[2]

        # Initialize hidden state to zeros (torch.zeros) of the right dimensions
        # This will be updated for each time step so no need to have a T dimension
        hidden = torch.zeros((B, self.hidden_size))

        # List of outputs for each time step
        outputs = []

        # Calculate hidden and output for each time step and update variables
        # Follow the equations in the problem set
        # torch.matmul is used for matrix multiplication
        # ----------------------------------------

        for t in range(T):
            hidden = self.tanh(hidden @ self.W_hh + x[:, t] @ self.W_xh + self.b_h)
            y = hidden @ self.W_hy + self.b_y
            outputs.append(y)

        # ----------------------------------------

        # Stack outputs along the time dimension to get a tensor of shape (B, T, output_size)
        # Hint: use torch.stack
        outputs = torch.stack(outputs, dim=1)

        return outputs

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs, dim=-1)
        return predictions


class RNN(nn.Module):
    def __init__(self, tokenizer, embedding_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: Initialize layers
        # Use nn.Embedding for the embedding layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # You can get the vocabulary size from the tokenizer with tokenizer.vocab_size
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_size)
        self.rnn_layer = RNNLayer(embedding_size, hidden_size, output_size)

    # TODO: Implement forward pass
    def forward(self, x):
        # Get text embedding
        x = self.embedding(x)

        # Pass through rnn layer
        x = self.rnn_layer(x)

        return x

