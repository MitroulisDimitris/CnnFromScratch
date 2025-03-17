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
        
    
    def forward(self,x):
        # prevent numerical instability by substracting the max value in each row
        x_exp = np.exp(x-np,max(x,axix=1,keepdimg=True))
        self.output = x_exp / np.sum(x_exp,axis=1,keepdims=True)
        
        return self.output
    
    def backwards(self,dout):
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
    def __init__(self, dropout_rate=0.5):
        assert 0 <= dropout_rate < 1, "Dropout rate must be between 0 and 1."
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, training=True):
        """
        Forward pass of Dropout.

        Parameters:
            x (np.ndarray): Input array of shape (batch_size, input_dim).
            training (bool): Whether the model is in training mode.

        Returns:
            np.ndarray: Output after applying dropout.
        """
        if training:
            # Create dropout mask: Keep neurons with probability (1 - dropout_rate)
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(x.dtype)
            return (x * self.mask) / (1.0 - self.dropout_rate)  # Scale during training
        else:
            return x  # No dropout during inference

    def backward(self, dout):
        """
        Backward pass of Dropout.

        Parameters:
            dout (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient with respect to the input.
        """
        return dout * self.mask / (1.0 - self.dropout_rate)  # Scale the gradients

        