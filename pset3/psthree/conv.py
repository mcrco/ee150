import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_size,
        padding=0,
        stride=1,
        filters=None,
        biases=None,
    ):
        super(Conv2d, self).__init__()

        # Number of input channels
        self.in_channels = in_channels

        # Number of filters
        self.out_channels = out_channels

        # Width/height of filters
        self.filter_size = filter_size

        # Padding for width and height
        self.padding = padding

        # Stride for width and height
        self.stride = stride

        self.filters = nn.Parameter(
            filters
            if filters is not None
            else torch.randn(out_channels, in_channels, filter_size, filter_size)
        )
        self.biases = nn.Parameter(
            biases if biases is not None else torch.randn(out_channels)
        )

    # TODO: Complete this function
    def convolution(self, x: torch.Tensor, filters: torch.Tensor):
        """
        2D convolution of filters over x
        x: [batch_size, channels, height, width]
        filters: [out_channels, in_channels, filter_size, filter_size]
        """

        # Batch size
        B = x.shape[0]

        # Number of input channels
        C_in = x.shape[1]

        # Height of input
        H = x.shape[2]

        # Width of input
        W = x.shape[3]

        # Number of output channels
        C_out = filters.shape[0]

        # Width/height of filter
        F = filters.shape[2]

        # Padding
        P = self.padding

        # Stride
        S = self.stride

        # Compute height of output considering padding and stride
        # Account for both even and odd dimensions
        H_out = (H + 2 * P - F) // S + 1

        # Compute width of output considering padding and stride
        # Account for both even and odd dimensions
        W_out = (W + 2 * P - F) // S + 1

        # Initialize output tensor with zeros
        # Use torch.zeros
        output = torch.zeros(B, C_out, H_out, W_out)

        # Compute padded_x to be x with P zero padding on all sides
        # Hint: Use torch.zeros and then set the appropriate elements to those of x
        padded_x = torch.zeros(B, C_in, H + 2 * P, W + 2 * P)
        padded_x[:, :, P : H + P, P : W + P] = x

        # Loop over all necessary dimensions to fill in the output tensor
        # To do this, you need to perform a 2d convolution of each filter over padded_x
        # Don't forget to account for the stride
        # Hint: You'll have 4 nested for loops
        # ----------------------------------------

        for ci in range(C_in):
            for co in range(C_out):
                for hi in range(H_out):
                    for wi in range(W_out):
                        patch = padded_x[
                            :, ci, hi * S : hi * S + F, wi * S : wi * S + F
                        ]
                        output[:, co, hi, wi] += torch.sum(
                            patch * filters[co, ci].unsqueeze(0), dim=(1, 2)
                        )

        # ----------------------------------------

        return output

    def forward(self, x: torch.Tensor):
        x = self.convolution(x, self.filters)
        return x + self.biases.view(1, -1, 1, 1)


class FasterConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        filter_size,
        padding=0,
        stride=1,
        filters=None,
        biases=None,
    ):
        super(FasterConv2d, self).__init__()

        # Number of input channels
        self.in_channels = in_channels

        # Number of filters
        self.out_channels = out_channels

        # Width/height of filters
        self.filter_size = filter_size

        # Padding for width and height
        self.padding = padding

        # Stride for width and height
        self.stride = stride

        self.filters = nn.Parameter(
            filters
            if filters is not None
            else torch.randn(out_channels, in_channels, filter_size, filter_size)
        )
        self.biases = nn.Parameter(
            biases if biases is not None else torch.randn(out_channels)
        )

    # TODO: Complete this function
    def convolution(self, x: torch.Tensor, filters: torch.Tensor):
        """
        2D convolution of filters over x
        x: [batch_size, channels, height, width]
        filters: [out_channels, in_channels, filter_size, filter_size]
        """

        # Batch size
        B = x.shape[0]

        # Number of input channels
        C_in = x.shape[1]

        # Height of input
        H = x.shape[2]

        # Width of input
        W = x.shape[3]

        # Number of output channels
        C_out = filters.shape[0]

        # Width/height of filter
        F = filters.shape[2]

        # Padding
        P = self.padding

        # Stride
        S = self.stride

        # Compute height of output considering padding and stride
        # Account for both even and odd dimensions
        H_out = (H + 2 * P - F) // S + 1

        # Compute width of output considering padding and stride
        # Account for both even and odd dimensions
        W_out = (W + 2 * P - F) // S + 1

        # Initialize output tensor with zeros
        # Use torch.zeros
        output = torch.zeros(B, C_out, H_out, W_out)

        # Compute padded_x to be x with P zero padding on all sides
        # Hint: Use torch.zeros and then set the appropriate elements to those of x
        padded_x = torch.zeros(B, C_in, H + 2 * P, W + 2 * P)
        padded_x[:, :, P : H + P, P : W + P] = x

        # If you think 4 nested for loops is computationally inefficient, you're right!
        # Let's do it faster, using broadcasting
        # If you've never heard of broadcasting, here's a good article to get acquainted: https://www.geeksforgeeks.org/understanding-broadcasting-in-pytorch/
        # For this implementation, you should only loop over H_out and W_out
        # Then, you'll need to update the shape of filters and the current patch from padded_x
        # so that you multiply every patch in the batch by every filter in filters
        # Ensure that you fill in the element of output accordingly
        # ----------------------------------------

        for hi in range(H_out):
            for wi in range(W_out):
                patch = padded_x[
                    :, :, hi * S : hi * S + F, wi * S : wi * S + F
                ].unsqueeze(dim=1)
                output[:, :, hi, wi] = torch.sum(
                    patch * filters,
                    dim=(2, 3, 4),
                )

        # ----------------------------------------

        return output

    def forward(self, x: torch.Tensor):
        x = self.convolution(x, self.filters)
        return x + self.biases.view(1, -1, 1, 1)
