import numpy as np

class Conv2D:
    def __init__(self, in_channels,out_channels,kernel_size, stride=1, padding=0, activation='relu',learning_rate=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if activation == 'relu':
            self.activation = activation
        self.learning_rate = learning_rate    
        
                
        # Xavier initialization
        self.weights = np.random.randn(out_channels,in_channels,kernel_size,kernel_size)* np.sqrt(1./(in_channels*kernel_size*kernel_size))
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
                        
                        output[b, oc, i, j] = np.sum(x_padded[b, :, h_start:h_end, w_start:w_end] * self.weights[oc]) + self.biases[oc]
    
        # ReLU Activation
        self.output = np.maximum(0, output)
        return self.output
        
    
    def backward(self, d_out, update_weights=True):
        batch_size, _, out_height, out_width = d_out.shape
        
        # Compute ReLU gradient without modifying d_out in place
        d_relu = d_out * (self.output > 0)
        
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
                        
                        d_w[oc] += x_padded[b, :, h_start:h_end, w_start:w_end] * d_relu[b, oc, i, j]
                        d_x[b, :, h_start:h_end, w_start:w_end] += self.weights[oc] * d_relu[b, oc, i, j]
                        d_b[oc] += d_relu[b, oc, i, j]
        
        if self.padding > 0:
            d_x = d_x[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        # If updating weights is desired, perform the update
        if update_weights:
            self.weights -= self.learning_rate * d_w
            self.biases -= self.learning_rate * d_b
        
        # Return gradients for gradient checking along with d_x
        return d_x, d_w, d_b


            

