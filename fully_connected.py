import numpy as np

class FullyConnectedLayer:
    
    def __init__(self,input_dim, output_dim, learning_rate=0.01):
        """
        Initializes the fully connected layer.
        
        Parameters:
            input_dim (int): Number of input features.
            output_dim (int): Number of output neurons.
            learning_rate (float): Learning rate for weight updates.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Xavier (Glorot) initialization for weights
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.bias = np.zeros((1, output_dim))
        
    def forward(self,X):
        self.input = X
        x_hat = np.dot(X, self.weights) + self.bias
        
        return x_hat
    
    def backward(self, dout):
        """
        Compute the gradients with respect to inputs, weights, and bias.
        Does not perform any parameter updates.

        Parameters:
            dout (np.ndarray): Gradient of the loss with respect to the output,
                            with shape (batch_size, output_dim).

        Returns:
            dx (np.ndarray): Gradient with respect to the input, shape (batch_size, input_dim)
        """
        # Compute gradients and store them in separate variables
        self.dweights = np.dot(self.input.T, dout)
        self.dbias = np.sum(dout, axis=0, keepdims=True)
        
        # Compute the gradient with respect to the input for backpropagation
        dx = np.dot(dout, self.weights.T)
    
        return dx

    def update_params(self):
        """
        Updates the weights and biases using gradient descent.
        """
        self.weights -= self.learning_rate * self.dweights
        self.bias -= self.learning_rate * self.dbias
