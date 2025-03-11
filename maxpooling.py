import numpy as np

class MaxPool:
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size  # pooling window size
        self.stride = stride            # step size between windows
        self.padding = padding          # number of zero-padding layers
        self.cache = {}                 # to store values for backward pass

    def forward(self, x):
        """
        Forward pass of max pooling.
        Parameters:
          x: Input tensor of shape (batch, channels, height, width)
        Returns:
          out: Output tensor after max pooling.
        """
        batch, channels, height, width = x.shape

        # Apply padding if needed.
        if self.padding > 0:
            x_padded = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            x_padded = x

        # Compute output dimensions.
        out_height = (x_padded.shape[2] - self.kernel_size) // self.stride + 1
        out_width  = (x_padded.shape[3] - self.kernel_size) // self.stride + 1

        out = np.zeros((batch, channels, out_height, out_width), dtype=x.dtype)
        mask = np.zeros_like(x_padded)

        for b in range(batch):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end   = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end   = w_start + self.kernel_size

                        window = x_padded[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        out[b, c, i, j] = max_val

                        # Create a mask: mark positions where the max value occurred.
                        window_mask = (window == max_val)
                        mask[b, c, h_start:h_end, w_start:w_end] += window_mask

        # Save variables for backward pass.
        self.cache['x_padded'] = x_padded
        self.cache['mask'] = mask
        self.cache['out_height'] = out_height
        self.cache['out_width'] = out_width

        return out

    def backward(self, d_out):
        """
        Backward pass for max pooling.
        Parameters:
          d_out: Upstream gradient of shape (batch, channels, out_height, out_width)
        Returns:
          d_x: Gradient with respect to the input (same shape as original x)
        """
        x_padded = self.cache['x_padded']
        mask = self.cache['mask']
        batch, channels, _, _ = x_padded.shape
        out_height = self.cache['out_height']
        out_width = self.cache['out_width']

        d_x_padded = np.zeros_like(x_padded)

        for b in range(batch):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end   = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end   = w_start + self.kernel_size

                        # Only the positions that were the max get the gradient.
                        d_x_padded[b, c, h_start:h_end, w_start:w_end] += \
                            mask[b, c, h_start:h_end, w_start:w_end] * d_out[b, c, i, j]

        # Remove padding if applied.
        if self.padding > 0:
            d_x = d_x_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_x = d_x_padded

        return d_x
