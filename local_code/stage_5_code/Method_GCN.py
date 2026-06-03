'''
Concrete MethodModule class for a semi-supervised Graph Convolutional Network MethodModule (GCN)
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_GCN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    # GCN semi-supervised nodes train efficiently across complete graph passes; 200 epochs is standard
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-2

    # it defines the GCN model architecture,
    # how many layers, feature input dimensions, internal representations, dropout limits, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, in_features=1433, hidden_dim=16, num_classes=7):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # First Graph Convolutional Layer weight projection parameters
        # Maps input continuous features into a compressed low-dimensional hidden embedding space
        self.gc_layer_1 = nn.Linear(in_features, hidden_dim)
        self.activation_func_1 = nn.ReLU()
        
        # Dropout structural element to prevent overfitting during tiny stratified mask optimizations
        self.dropout_layer = nn.Dropout(p=0.5)
        
        # Second Graph Convolutional Layer weight projection parameters
        # Projects the mixed neighborhood message representation directly into target multi-class categories
        self.gc_layer_2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, adj):
        '''Forward propagation implementing structural neighborhood aggregate transformations'''
        # Propagate the first linear map and multiply by the normalized adjacency matrix structure
        # Graph Operation formula: Z = A_norm * X * W_1
        h = self.gc_layer_1(x)
        h = torch.spmm(adj, h)
        h = self.activation_func_1(h)
        h = self.dropout_layer(h)
        
        # Propagate the second linear map and perform final structural context aggregation
        # Output layer formula: Logits = A_norm * H * W_2
        h = self.gc_layer_2(h)
        y_pred = torch.spmm(adj, h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self, graph_data, split_data):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()

        # For the plot
        loss_history = []
        
        # Extract the monolithic graph feature structures and target classification index vectors
        X = graph_data['X']
        y = graph_data['y']
        adj = graph_data['utility']['A']
        
        # Extract train and validation positioning index slices from data partitions
        idx_train = split_data['idx_train']
        idx_val = split_data['idx_val']

        # Graph learning updates the complete network simultaneously; processing operates over full-graph cycles
        for epoch in range(self.max_epoch):
            self.train()
            
            # get the output through forward propagation across the absolute complete network space
            y_pred = self.forward(X, adj)
            
            # calculate the training loss exclusively over the allocated stratified training nodes mask
            train_loss = loss_function(y_pred[idx_train], y[idx_train])

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # update the variables according to the optimizer and the gradients
            optimizer.step()

            # Explicitly isolate scalar float numbers via .item() to preserve memory overhead
            loss_history.append(train_loss.item())

            if epoch % 20 == 0:
                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(X, adj)
                    pred_labels = val_outputs[idx_train].max(1)[1]
                    acc = (pred_labels == y[idx_train]).float().mean()
                    print(f"Epoch {epoch:03d} | Train Mask Acc: {acc.item():.4f} | Full-Graph Loss: {train_loss.item():.4f}")
            
        return loss_history
    
    def test(self, graph_data, split_data):
        self.eval()
        
        X = graph_data['X']
        adj = graph_data['utility']['A']
        idx_test = split_data['idx_test']
        
        # disable gradient calculation for efficiency during testing
        with torch.no_grad():
            y_pred_full = self.forward(X, adj)
            # Isolate predictions belonging exclusively to testing matrix coordinates
            test_predictions = y_pred_full[idx_test].max(1)[1].cpu().numpy()
                
        self.train() # set back to train mode after testing
        return test_predictions
    
    def run(self):
        print('method running...')
        print('--start training...')
        # Map parameters straight out of the instructor's graph structural schema payload shapes
        loss_history = self.train_model(self.data['graph'], self.data['train_test_val'])
        print('--start testing...')
        pred_y = self.test(self.data['graph'], self.data['train_test_val'])
        
        # Fetch the baseline labels corresponding to your active evaluation targets
        test_true_y = self.data['graph']['y'][self.data['train_test_val']['idx_test']].cpu().numpy()
        return {'pred_y': pred_y, 'true_y': test_true_y, 'loss_history': loss_history}