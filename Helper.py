import numpy as np
class ReLU:
    def forward(self, x):
        self.x = x  # Save for backward pass
        return np.maximum(0, x)
    
    def backward(self, dout):
        # Derivative: 1 for x > 0, 0 otherwise
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx
    

class SoftMax:
    def __init__(self):
        self.output = None
          
    def forward(self, x):
        # Prevent numerical instability by subtracting the max value in each row
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = x_exp / np.sum(x_exp, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dout):
        batch_size, num_classes = self.output.shape
        dx = np.zeros_like(self.output)

        for i in range(batch_size):
            # Compute the Jacobian matrix of softmax
            softmax_i = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(softmax_i) - np.dot(softmax_i, softmax_i.T)

            # Compute gradient: dout * Jacobian
            dx[i] = np.dot(jacobian, dout[i])

        return dx
 
class Dropout:
    def __init__(self, dropout_rate=0.2):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(x.dtype) / (1.0 - self.dropout_rate)
            return x * self.mask
        return x  # No dropout during inference

    def backward(self, dout):
        if self.mask is None:
            raise ValueError("Dropout mask was not set. Ensure forward pass is called before backward.")
        return dout * self.mask  # Ensure shapes match correctly

class Flatten:
    def __init__(self):
        self.input_shape = None
        
    def forward(self, x, training=True):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output):
        # Reshape grad_output to the original input shape and return it.
        return grad_output.reshape(self.input_shape)