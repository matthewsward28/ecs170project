'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np

class Dataset_Loader(dataset):
    dataset_source_folder_path = None
    train_file_name = None
    test_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading stage 2 data...')
        # Load train and test files
        train_data = self.load_csv(self.train_file_name)
        test_data = self.load_csv(self.test_file_name)
        return {'train': train_data, 'test': test_data}

    def load_csv(self, file_name):
        X = []
        y = []
        full_path = self.dataset_source_folder_path + file_name
        
        with open(full_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                if not line: continue
                
                # Comma seperated
                elements = [int(i) for i in line.split(',')]
                
                # First element is label, rest are features
                y.append(elements[0])
                X.append(elements[1:])
        
        # Normalizing feature values for MLP
        X = np.array(X) / 255.0
        y = np.array(y)
        
        return {'X': X, 'y': y}