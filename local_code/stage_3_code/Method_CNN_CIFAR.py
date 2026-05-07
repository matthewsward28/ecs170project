'''
Concrete MethodModule class for a specific learning MethodModule
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    # CIFAR is more complex, so we use 100 epochs to allow convergence
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the CNN model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # Conv Block 1: Input 3 channels (RGB), output 32 filters, 3x3 kernel
        # check here for nn.Conv2d doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv_column_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 32x32 -> 16x16
        )
        
        # Conv Block 2: Input 32 filters, output 64 filters, 3x3 kernel
        self.conv_column_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 16x16 -> 8x8
        )

        # Final Fully Connected layers
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # 64 filters * 8 * 8 pixels = 4096 input features
        self.fc_layer_1 = nn.Linear(64 * 8 * 8, 512)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(512, 10)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # convolution layers
        h = self.conv_column_1(x)
        h = self.conv_column_2(h)
        
        # flatten the output for the fully connected layer
        # h.size(0) denotes the batch size
        h = h.view(h.size(0), -1)
        
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(h))
        
        # output layer result
        # we return raw logits, CrossEntropyLoss will handle normalized probability distributions
        y_pred = self.fc_layer_2(h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()

        # it defines the size of the mini-batch
        batch_size = 64
        # For the plot
        loss_history = []

        # Convert input to tensors for pytorch operation
        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        num_samples = X_tensor.size(0)

        # it will be an iterative gradient updating process using mini-batches
        for epoch in range(self.max_epoch):
            # shuffle the data indices for each epoch to avoid bias
            indices = torch.randperm(num_samples)
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            
            epoch_loss = 0.0

            # mini-batch iteration
            for i in range(0, num_samples, batch_size):
                # slice the input into smaller batches
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # get the output through forward propagation
                y_pred = self.forward(X_batch)
                # calculate the training loss for the current batch
                train_loss = loss_function(y_pred, y_batch)

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # update the variables according to the optimizer and the gradients
                optimizer.step()

                epoch_loss += train_loss.item()

            # Record the average loss of all batches for plotting
            avg_loss = epoch_loss / (num_samples / batch_size)
            loss_history.append(avg_loss)

            if epoch % 10 == 0:
                # Calculate accuracy on the last batch of the epoch for tracking
                pred_labels = y_pred.max(1)[1]
                acc = (pred_labels == y_batch).float().mean()
                print(f"Epoch {epoch} | Batch Acc: {acc.item():.4f} | Avg Loss: {avg_loss:.4f}")
            
        return loss_history
    
    def test(self, X):
        # do the testing, and result the result
        # disable gradient calculation for efficiency during testing
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}