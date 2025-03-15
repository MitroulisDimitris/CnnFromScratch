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