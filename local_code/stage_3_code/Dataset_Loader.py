'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle
import numpy as np
import torch

class Dataset_Loader(dataset):
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = None
        self.dataset_file_name = None

    def load(self):
        full_path = self.dataset_source_folder_path + self.dataset_file_name
        
        # load the pickle file
        with open(full_path, 'rb') as f:
            data = pickle.load(f)
            
        # process train and test sets
        train_data = self.process_data(data['train'])
        test_data = self.process_data(data['test'])
        
        return {'train': train_data, 'test': test_data}

    def process_data(self, data_list):
        X = []
        y = []
        
        for instance in data_list:
            # image_matrix is already a 2D or 3D numpy array from the pickle
            image = np.array(instance['image'])
            label = instance['label']
            
            # normalize pixels to [0, 1]
            image = image / 255.0
            
            # adjust labels for ORL
            if self.dataset_name == 'ORL' and label > 0:
                label -= 1
                
            X.append(image)
            y.append(label)
            
        X = np.array(X)
        y = np.array(y)
        
        # Handle Channels [Batch, Channel, Height, Width]
        # MNIST is (N, 28, 28) -> needs to be (N, 1, 28, 28)
        # CIFAR/ORL is (N, H, W, 3) -> needs to be (N, 3, H, W)
        if len(X.shape) == 3: # Grayscale (MNIST)
            X = np.expand_dims(X, axis=1)
        elif len(X.shape) == 4: # Color (CIFAR/ORL)
            # PyTorch expects (Channels, Height, Width), but Pickle usually has (Height, Width, Channels)
            X = np.transpose(X, (0, 3, 1, 2))
            
            if self.dataset_name == 'ORL':
                X = X[:, 0:1, :, :] # Keep only R channel
                
        return {'X': torch.FloatTensor(X), 'y': torch.LongTensor(y)}