import numpy as np

class Conv2D:
    def __init__(self, in_channels,out_channels,kernel_size, stride=1, padding=0, activation='relu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if activation == 'relu':
            self.activation = activation
            
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
            
            out_height = (input_height-kernel_size+2*self.padding) // self.stride+1
            out_width = (input_width-kernel_size+2*self.padding) // self.stride+1
            
            output = np.zeros((batch_size,self.out_channels,out_height,out_width))
            
            for b in range(batch_size):
                for oc in range(self.out_channels):
                    for i in range(out_height):
                        for j in range(out_width):
                            h_start = i*self.stride
                            h_end = h_start+self.kernel_size
                            w_start = j*self.stride
                            w_end = w_start+self.kernel_s0ze
                            
                            output[b, oc, i, j] = np.sum(x_padded[b, :, h_start:h_end, w_start:w_end] * self.weights[oc]) + self.biases[oc]
        
            # ReLU Activation
            self.output = np.maximum(0, output)
            return self.output
            
        
        #def backwards(self, input):
            

