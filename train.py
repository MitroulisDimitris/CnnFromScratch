from convolution import Conv2D
from maxpooling import MaxPool
from fully_connected import FullyConnectedLayer
from Helper import ReLU, SoftMax, Dropout, Flatten
import numpy as np

class Model:
    def __init__(self,lr=0.01):
        self.conv1 = Conv2D(in_channels=1,out_channels=32,kernel_size=3,learning_rate=lr)
        self.relu1 = ReLU()
        self.pool1 = MaxPool(kernel_size=2,stride=2)


        self.conv2 = Conv2D(in_channels=32,out_channels=64,kernel_size=3,learning_rate=lr)
        self.relu2 = ReLU()
        self.pool2 = MaxPool(kernel_size=2,stride=2)

        self.flatten = Flatten()

        self.fc1= FullyConnectedLayer(input_dim=5*5*64,output_dim=128,learning_rate=lr)
        self.relu3 = ReLU()

        self.dropout = Dropout(dropout_rate=0.2)
        self.fc2= FullyConnectedLayer(input_dim=128,output_dim=10,learning_rate=lr)

        self.softmax = SoftMax()

        self.layers = [
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.flatten,
            self.fc1,
            self.relu3,
            self.dropout,    
            self.fc2,
            self.softmax
        ]
             
    def forward(self, x, training=True):
        out = x
        for layer in self.layers:
            if isinstance(layer, Dropout):  # Ensure dropout behavior is controlled
                out = layer.forward(out, training=training)
            else:
                out = layer.forward(out)
        return out

    
    def backward(self,d_out):
        """
        Backward pass through all layers (reverse order).
        :param d_out: Gradient of the loss w.r.t. the output of the last layer
        :return: Gradient w.r.t. the input of the first layer (not always needed)
        """
        
        grad= d_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
        
    def compute_loss(self, preds, labels):
        """
        Cross-entropy loss.
        :param preds: Predictions after softmax, shape (batch_size, 10)
        :param labels: One-hot encoded true labels, shape (batch_size, 10)
        :return: Scalar loss
        """
        # To avoid numerical instability, clip predictions
        eps = 1e-9
        preds_clipped = np.clip(preds, eps, 1 - eps)

        # Cross-entropy
        batch_size = preds.shape[0]
        loss = -np.sum(labels * np.log(preds_clipped)) / batch_size
        return loss

    def train_on_batch(self, x_batch, y_batch):
        """
        Single training step on a batch of data:
        1) Forward pass
        2) Compute loss
        3) Backward pass
        4) Update parameters
        :param x_batch: (batch_size, 1, 28, 28)
        :param y_batch: (batch_size, 10) one-hot
        :return: Scalar loss
        """
        # 1) Forward pass (training=True enables dropout)
        preds = self.forward(x_batch, training=True)
        
        # 2) Compute loss
        loss = self.compute_loss(preds, y_batch)
        
        # 3) Compute gradient of the loss w.r.t. the final layer’s output

        batch_size = y_batch.shape[0]
        d_out = (preds - y_batch) / batch_size # for softmax + cross-entropy
        
        # 4) Backward pass
        self.backward(d_out)
        
        # 5) Update parameters in each layer that has update_weights()
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights()
        
        return loss

    def predict(self, x):
        """
        Inference (forward pass without dropout).
        :param x: (batch_size, 1, 28, 28)
        :return: Class probabilities, shape (batch_size, 10)
        """
        probs = self.forward(x, training=False)
        return probs

    def evaluate_accuracy(self, x, y_true):
        """
        Compute accuracy on a batch of data.
        :param x: (batch_size, 1, 28, 28)
        :param y_true: (batch_size, 10) one-hot
        :return: accuracy (float)
        """
        probs = self.predict(x)  # shape: (batch_size, 10)
        y_pred = np.argmax(probs, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(y_pred == y_true_labels)
        return accuracy