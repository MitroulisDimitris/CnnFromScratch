import numpy as np

class FullyConnectedLayer:
    
    def __init__(self,input_dim, output_dim, learning_rate=0.01, activation='relu'):
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
        
        if activation == 'relu':
            self.activation = activation
            
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
            tuple: A tuple containing:
                - dx (np.ndarray): Gradient with respect to the input, shape (batch_size, input_dim)
                - dweights (np.ndarray): Gradient with respect to the weights, shape (input_dim, output_dim)
                - dbias (np.ndarray): Gradient with respect to the bias, shape (1, output_dim)
        """
        self.dweights = np.dot(self.input.T, dout)
        self.dbias = np.sum(dout, axis=0, keepdims=True)
        dx=np.dot(dout,self.weights.T)
        
        return dx, self.dweights,self.dbias