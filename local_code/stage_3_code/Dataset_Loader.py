'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle
import numpy as np
import torch
import os

class Dataset_Loader(dataset):
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        # initialize with empty strings instead of None to prevent concatenation errors
        self.dataset_source_folder_path = ""
        self.dataset_file_name = dName if dName else ""
        self.channel = 1
        self.height = 28
        self.width = 28

    def load(self):
        # use os.path.join for safer path concatenation
        # this joins './data/stage_3_data/' and 'MNIST' correctly
        full_path = os.path.join(self.dataset_source_folder_path, self.dataset_file_name)
        
        print(f"Attempting to load: {full_path}")
        
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
            image = np.array(instance['image'])
            label = instance['label']
            
            # Normalize
            image = image / 255.0
            
            # label adjustment for ORL
            if self.dataset_name == 'ORL' and label > 0:
                label -= 1
                
            X.append(image)
            y.append(label)
            
        X = np.array(X)
        y = np.array(y)
        
        # dimension handling
        if len(X.shape) == 3: # (N, H, W) -> (N, 1, H, W)
            X = np.expand_dims(X, axis=1)
        elif len(X.shape) == 4: # (N, H, W, C) -> (N, C, H, W)
            X = np.transpose(X, (0, 3, 1, 2))
            
            if self.dataset_name == 'ORL':
                X = X[:, 0:1, :, :]
                
        return {'X': torch.FloatTensor(X), 'y': torch.LongTensor(y)}