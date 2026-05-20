'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import torch 

class Evaluate_Metrics(evaluate):
    def __init__(self, eName=None, eDescription=None):
        super().__init__(eName, eDescription)

    def evaluate(self, pred_y, true_y):
        # if the input is a torch Tensor, move to CPU and convert to numpy
        if isinstance(pred_y, torch.Tensor):
            pred_y = pred_y.cpu().numpy()
        if isinstance(true_y, torch.Tensor):
            true_y = true_y.cpu().numpy()
            
        # standard numpy conversion for safety
        if not isinstance(pred_y, np.ndarray):
            pred_y = np.array(pred_y)
        if not isinstance(true_y, np.ndarray):
            true_y = np.array(true_y)

        # calculate metrics
        accuracy = accuracy_score(true_y, pred_y)
        precision = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        recall = recall_score(true_y, pred_y, average='weighted', zero_division=0)
        f1 = f1_score(true_y, pred_y, average='weighted', zero_division=0)

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }