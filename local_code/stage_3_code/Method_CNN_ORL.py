'''
Concrete MethodModule class for a specific learning MethodModule
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_CNN_ORL(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    # ORL is small, so it converges quickly, but we need to watch for overfitting
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4

    # it defines the the CNN model architecture, e.g.,
    # how many layers, kernel size, pooling, activation function, etc.
    # the size of the input/output portal should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # Conv Layer 1: Input 1 channel (gray), output 16 filters, 5x5 kernel
        # check here for nn.Conv2d doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 112x92 -> 56x46
        )
        
        # Conv Layer 2: Input 16 filters, output 32 filters, 5x5 kernel
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2) # 56x46 -> 28x23
        )
        
        # Dropout to prevent overfitting on the small ORL dataset
        self.dropout = nn.Dropout(p=0.5)
        
        # Output layer: fully connected layer
        # 32 filters * 28 * 23 pixels = 20608 input features
        self.fc_layer = nn.Linear(32 * 28 * 23, 40)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x):
        '''Forward propagation'''
        # Convolutional layers
        h = self.conv_layer_1(x)
        h = self.conv_layer_2(h)
        
        # flatten the output for the fully connected layer
        # h.size(0) is the batch size
        h = h.view(h.size(0), -1)
        
        # apply dropout before the final layer
        h = self.dropout(h)
        
        # output layer result: 40 classes for the 40 people in ORL
        y_pred = self.fc_layer(h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()

        # ORL is small, so we use a smaller batch size
        batch_size = 10
        
        # For the plot
        loss_history = []

        # Convert input to tensors for pytorch operation
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        num_samples = X_tensor.size(0)

        # it will be an iterative gradient updating process using mini-batches
        for epoch in range(self.max_epoch): 
            # shuffle data indices
            indices = torch.randperm(num_samples)
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            
            epoch_loss = 0.0

            # mini-batch iteration
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # get output through forward propagation
                y_pred = self.forward(X_batch)
                # calculate training loss
                train_loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                epoch_loss += train_loss.item()
            
            # Record average loss for the plot
            avg_loss = epoch_loss / (num_samples / batch_size)
            loss_history.append(avg_loss)

            if epoch % 10 == 0:
                pred_labels = y_pred.max(1)[1]
                acc = (pred_labels == y_batch).float().mean()
                print(f"Epoch {epoch} | Batch Acc: {acc.item():.4f} | Avg Loss: {avg_loss:.4f}")
            
        return loss_history
    
    def test(self, X):
        # disable gradient and dropout during testing
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
        self.train() # set back to train mode
        return y_pred.max(1)[1]