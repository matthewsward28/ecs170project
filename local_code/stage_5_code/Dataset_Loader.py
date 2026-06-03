'''
Concrete IO class for graph network datasets handling node classifications
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from local_code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp

class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, seed=None, dName=None, dDescription=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)

    def adj_normalize(self, mx):
        '''Apply symmetric degree normalization to structural adjacency matrices'''
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        '''Convert a compressed SciPy sparse matrix into a structured PyTorch sparse tensor representation'''
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        '''Encode textual string categories into absolute multi-class one-hot binary matrices'''
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        onehot_labels = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return onehot_labels

    def load(self):
        '''Main loader driver reading node features, mapping link topological targets, and applying stratified random partitions'''
        print('Loading {} dataset...'.format(self.dataset_name))

        # Load node attributes from data file
        # Matrix Layout: <node_id> <continuous_features> <string_class_label>
        idx_features_labels = np.genfromtxt("{}/node".format(self.dataset_source_folder_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        onehot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # Isolate and register node index positions to form clean lookup mapping dictionaries
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        reverse_idx_map = {i: j for i, j in enumerate(idx)}
        
        # Load structural graph connections from link file
        # Layout: A B, signifying directed message passing pointing from B to A
        edges_unordered = np.genfromtxt("{}/link".format(self.dataset_source_folder_path), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        
        # Construct the raw structural sparse adjacency coordinate representation matrix
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(onehot_labels.shape[0], onehot_labels.shape[0]), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        # Inject self-loops (A + I) and apply structural normalization computations to smooth gradient propagation steps
        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        # Convert continuous array representations into final active PyTorch tensor elements
        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(onehot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # Lock random seeds to ensure dataset partition and slicing execution steps are reproducible
        np.random.seed(42) 
        labels_np = labels.numpy()
        num_classes = onehot_labels.shape[1]
        
        # Configure evaluation partition distribution quotas to match explicit assignment constraints
        if self.dataset_name == 'cora':
            train_per_class = 20
            test_per_class = 150
        elif self.dataset_name == 'citeseer':
            train_per_class = 20
            test_per_class = 200
        elif self.dataset_name == 'pubmed':
            train_per_class = 20
            test_per_class = 200
        else:
            train_per_class = 5
            test_per_class = 5

        idx_train_list = []
        idx_test_list = []
        idx_val_list = [] 

        # Iterate through unique categories to harvest uniform, balanced structural node counts
        for c in range(num_classes):
            # Isolate rows belonging exclusively to class index identifier c
            class_indices = np.where(labels_np == c)[0]
            np.random.shuffle(class_indices)
            
            # Slice strict stratified quotas out of the localized shuffled pools
            train_nodes = class_indices[:train_per_class]
            test_nodes = class_indices[train_per_class : train_per_class + test_per_class]
            val_nodes = class_indices[train_per_class + test_per_class : train_per_class + test_per_class + 10] 
            
            idx_train_list.extend(train_nodes)
            idx_test_list.extend(test_nodes)
            idx_val_list.extend(val_nodes)

        # Shuffle complete lists to clear grouped block layouts before building long indexing tensors
        np.random.shuffle(idx_train_list)
        np.random.shuffle(idx_test_list)
        np.random.shuffle(idx_val_list)

        idx_train = torch.LongTensor(idx_train_list)
        idx_test = torch.LongTensor(idx_test_list)
        idx_val = torch.LongTensor(idx_val_list)

        print(f"Stratified Partition Generated -> Train Nodes: {len(idx_train)} | Test Nodes: {len(idx_test)} | Val Nodes: {len(idx_val)}")

        # Package data dictionary components into final graph pipeline object states
        train_test_val = {'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        graph = {'node': idx_map, 'edge': edges, 'X': features, 'y': labels, 'utility': {'A': adj, 'reverse_idx': reverse_idx_map}}
        return {'graph': graph, 'train_test_val': train_test_val}