import torch
import torch.nn as nn
import numpy as np
import math


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()

        # d_model = d_k = d_v
        self.d_model = d_model

        # TODO: Initialize weights and biases
        # Instead of using nn.Parameter, use nn.Linear (for speed)
        # Don't need any special initializaton
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # TODO: Initialize softmax
        # In order to compute attention scores, what dimension is softmax applied to?
        # Hint: You want each row of your attention scores to be a probability distribution
        # Use nn.Softmax
        self.softmax = nn.Softmax(dim=-1)

    # TODO: Implement forward pass
    # Follow the formula you wrote in the problem set
    # Note that self-attention is applied to every input embedding in x individually
    # Thus, matmuls are broadcasted across the batch and only the last two dimensions matter
    # To transpose K use torch.Tensor.transpose on the two dimensions you want to swap
    # Keep in mind that linear layers are only applied on the last dimension of x by default
    # Fun fact: Implementing this method is a real interview question for Machine Learning Engineering roles
    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        logits = Q @ K.transpose(-1, -2) / math.sqrt(self.d_model)
        output = self.softmax(logits) @ V
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads

        # TODO: Initialize weights and biases
        # Multi-head attention can be implemented by first applying the same linear layers as in self attention
        # Then, splitting them into different heads and computing attention scores for each head individually
        # Finally, concatenating the heads together and applying the output linear layer
        # Instead of using nn.Parameter, use nn.Linear (for speed)
        # Don't need any special initializaton
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # TODO: Define d_k
        # d_k is only used when applying attention for each head individually
        self.d_k = d_model // num_heads

        # TODO: Initialize softmax
        # In order to compute attention scores, what dimension is softmax applied to?
        # Hint: You want each row of your attention scores to be a probability distribution
        # Use nn.Softmax
        self.softmax = nn.Softmax(dim=-1)

    # TODO: Implement forward pass
    # Follow these steps
    # 1) Apply the linear transformations of q, k, v
    # 2) Use torch.Tensor.reshape to reshape each one to split each vector into num_heads parts: (B, T, num_heads, d_K)
    #    This will effectively split each embedding vector into num_heads parts
    # 3) Use torch.Tensor.transpose to achieve the shape (B, num_heads, T, d_k)
    # 4) Compute the attention scores for each head
    #    Notice that in SelfAttention, the batch dimension was ignored
    #    With the same code, both the batch and num_heads dimensions are ignored
    #    This way you are effectively computing attention for each head individually
    # 5) Use torch.Tensor.transpose to achieve the shape (B, T, num_heads, d_k)
    # 6) Use torch.Tensor.reshape to effectively concatenate the heads together
    # 7) Apply the final linear transformation
    def forward(self, x):
        B, T = x.shape[:-1]
        Q = self.W_q(x).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        logits = Q @ K.transpose(-1, -2) / math.sqrt(self.d_model)
        attn = (self.softmax(logits) @ V).transpose(1, 2).reshape(B, T, self.d_model)
        output = self.W_o(attn)
        return output
