import numpy as np

class Conv2D:
    def __init__(self, in_channels,out_channels,kernel_size,learning_rate=0.01, stride=1, padding=0,activation='relu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if activation == 'relu':
            self.activation = activation
        self.learning_rate = learning_rate    
        
                
        # Xavier initialization
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        limit = np.sqrt(2 / (fan_in + fan_out))

        self.weights = np.random.uniform(-limit, limit, 
                                         (out_channels, in_channels, kernel_size, kernel_size))
        self.biases = np.zeros((out_channels,1))
        
    def pad_input(self,x):
        if self.padding>0:
            return np.pad(x,((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode = 'constant')
        return x
    
    def forward(self, x):
        self.input = x
        x_padded = self.pad_input(x)
        batch_size, _ , input_height, input_width = x.shape
        
        out_height = (input_height-self.kernel_size+2*self.padding) // self.stride+1
        out_width = (input_width-self.kernel_size+2*self.padding) // self.stride+1
        
        output = np.zeros((batch_size,self.out_channels,out_height,out_width))
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i*self.stride
                        h_end = h_start+self.kernel_size
                        w_start = j*self.stride
                        w_end = w_start+self.kernel_size
                        
        output[b, oc, i, j] = np.sum(x_padded[b, :, h_start:h_end, w_start:w_end] * self.weights[oc]) + self.biases[oc].item()
    
        # ReLU Activation
        self.output = np.maximum(0, output)
        return self.output
      
    def backward(self, d_out):
        """
        Computes gradients with respect to the input, weights, and biases.
        This backward function does not incorporate the ReLU derivative and does not update parameters.
        
        Parameters:
            d_out (np.ndarray): Gradient of the loss with respect to the output of the layer,
                                with shape (batch_size, out_channels, out_height, out_width)
                                
        Returns:
            tuple: (d_x, d_w, d_b) where:
                - d_x is the gradient with respect to the input,
                - d_w is the gradient with respect to the weights,
                - d_b is the gradient with respect to the biases.
        """
        batch_size, _, out_height, out_width = d_out.shape
        
        # Assume d_out is already the appropriate gradient (e.g., the activation derivative has been applied externally)
        x_padded = self.pad_input(self.input)
        d_x = np.zeros_like(x_padded)
        d_w = np.zeros_like(self.weights)
        d_b = np.zeros_like(self.biases)
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        d_w[oc] += x_padded[b, :, h_start:h_end, w_start:w_end] * d_out[b, oc, i, j]
                        d_x[b, :, h_start:h_end, w_start:w_end] += self.weights[oc] * d_out[b, oc, i, j]
                        d_b[oc] += d_out[b, oc, i, j]
        
        if self.padding > 0:
            d_x = d_x[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return d_x


    def update_params(self,d_w,d_b):
        """
        Updates the weights and biases using gradient descent.
        """
        self.weights -= self.learning_rate * d_w
        self.biases -= self.learning_rate * d_b
    

