'''
Concrete MethodModule class for a specific sequential learning MethodModule
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_RNN_Text(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    # text datasets are complex, 20 epochs provides a clear look at optimization paths
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the RNN model architecture,
    # how many layers, embedding size, recurrent hidden dimensions, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, vocab_size=15000, embedding_dim=128, hidden_dim=128):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # Word Embedding Layer
        # Maps word indices into a dense, continuous vector space
        # check here for nn.Embedding doc: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # Vanilla Recurrent Neural Network Block
        # Processes the sequential elements step-by-step
        # batch_first=True expects matrix dimensions organized as [Batch, Sequence_Length, Embedding_Dim]
        # check here for nn.RNN doc: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn_layer = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Final Fully Connected layers for sentiment classification
        # Projects the recurrent hidden representations into binary categories (Negative vs Positive)
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(hidden_dim, 64)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(64, 2)

    def forward(self, x):
        '''Forward propagation'''
        # Convert index integer tokens to dense feature coordinates
        # output shape: [Batch Size, Sequence Length, Embedding Dim]
        embedded = self.embedding_layer(x)
        
        # Propagate embeddings through sequential recurrent hidden elements
        # rnn_out shape: [Batch Size, Sequence Length, Hidden Dim]
        # hidden_n shape: [1, Batch Size, Hidden Dim] representing final temporal sequence states
        rnn_out, hidden_n = self.rnn_layer(embedded)
        
        # Isolate the final step hidden states to summarize information from entire sentences
        # Shape transforms from [1, Batch, Hidden] to [Batch, Hidden]
        h = hidden_n.squeeze(0)
        
        # Fully connected projection layers
        h = self.activation_func_1(self.fc_layer_1(h))
        
        # output layer result
        # we return raw logits, CrossEntropyLoss will handle normalized probability distributions
        y_pred = self.fc_layer_2(h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()

        # it defines the size of the mini-batch
        batch_size = 64
        # For the plot
        loss_history = []

        # Convert input matrix data sequences to integer LongTensors for text index retrieval
        X_tensor = torch.LongTensor(np.array(X))
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

            if epoch % 5 == 0:
                # Calculate accuracy on the last batch of the epoch for tracking
                pred_labels = y_pred.max(1)[1]
                acc = (pred_labels == y_batch).float().mean()
                print(f"Epoch {epoch} | Batch Acc: {acc.item():.4f} | Avg Loss: {avg_loss:.4f}")
            
        return loss_history
    
    def test(self, X):
        self.eval()
        # do the testing, and result the result
        # disable gradient calculation for efficiency during testing
        with torch.no_grad():
            y_pred = self.forward(torch.LongTensor(np.array(X)))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        self.train() # set back to train mode after testing
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}