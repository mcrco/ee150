from math import sqrt
import torch
import torch.nn as nn
import numpy as np


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMLayer, self).__init__()

        self.hidden_size = hidden_size

        # TODO: Initialize weights and biases
        # Initialize weights with Xavier initialization
        # For weight matrix of shape (n_in, n_out):
        # Normal distribution with mean 0 and variance 2 / (n_in + n_out)
        # Use torch.randn
        # Initialize biases with zeros (torch.zeros)
        # Use nn.Parameter to create the weights and biases and pass in the initialziation

        # Input gate weights
        self.W_xi = torch.randn(input_size, hidden_size) * sqrt(
            2 / (input_size + hidden_size)
        )
        self.W_hi = torch.randn(hidden_size, hidden_size) * sqrt(1 / hidden_size)
        self.b_i = torch.zeros(hidden_size)

        # Forget gate weights
        self.W_xf = torch.randn(input_size, hidden_size) * sqrt(
            2 / (input_size + hidden_size)
        )

        self.W_hf = torch.randn(hidden_size, hidden_size) * sqrt(1 / hidden_size)
        self.b_f = torch.zeros(hidden_size)

        # Output gate weights
        self.W_xo = torch.randn(input_size, hidden_size) * sqrt(
            2 / (input_size + hidden_size)
        )

        self.W_ho = torch.randn(hidden_size, hidden_size) * sqrt(1 / hidden_size)
        self.b_o = torch.zeros(hidden_size)

        # Cell state weights
        self.W_xc = torch.randn(input_size, hidden_size) * sqrt(
            2 / (input_size + hidden_size)
        )

        self.W_hc = torch.randn(hidden_size, hidden_size) * sqrt(1 / hidden_size)
        self.b_c = torch.zeros(hidden_size)

        # Output weights
        self.W_hy = torch.randn(hidden_size, output_size) * sqrt(
            2 / (hidden_size + output_size)
        )
        self.b_y = torch.zeros(output_size)

        # TODO: Initialize activation functions
        # Use nn.Tanh, nn.Sigmoid
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    # TODO: Implement forward pass
    def forward(self, x):
        # Define dimensions
        B = x.shape[0]
        T = x.shape[1]
        H = x.shape[2]

        # Initialize h_t and c_t to zeros (torch.zeros) of the right dimensions
        # This will be updated for each time step so no need to have a T dimension
        h_t = torch.zeros((B, self.hidden_size))
        c_t = torch.zeros((B, self.hidden_size))

        # List of outputs for each time step
        outputs = []

        # Calculate h_t, c_t, and output for each time step and update variables
        # Follow the equations in the problem set
        # Element-wise multiplcation can be done with just "*"
        # torch.matmul is used for matrix multiplication
        # ----------------------------------------

        for t in range(T):
            f_t = self.sigmoid(x[:, t] @ self.W_xf + h_t @ self.W_hf + self.b_f)
            i_t = self.sigmoid(x[:, t] @ self.W_xi + h_t @ self.W_hi + self.b_i)
            c_t_ = self.tanh(x[:, t] @ self.W_xc + h_t @ self.W_hc + self.b_c)
            c_t = f_t * c_t + i_t * c_t_

            o_t = self.sigmoid(x[:, t] @ self.W_xo + h_t @ self.W_ho + self.b_o)
            h_t = o_t * self.tanh(c_t)

            y_t = h_t @ self.W_hy + self.b_y
            outputs.append(y_t)

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


class LSTM(nn.Module):
    def __init__(self, tokenizer, embedding_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: Initialize layers
        # Use nn.Embedding for the embedding layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # You can get the vocabulary size from the tokenizer with tokenizer.vocab_size
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_size)
        self.lstm_layer = LSTMLayer(embedding_size, hidden_size, output_size)

    # TODO: Implement forward pass
    def forward(self, x):
        # Get text embedding
        x = self.embedding(x)

        # Pass through lstm layer
        x = self.lstm_layer(x)

        return x

