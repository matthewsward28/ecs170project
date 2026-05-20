'''
Main orchestration execution script for Stage 4 text classification tasks
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Text import Method_RNN_Text
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Train_Test_Load import Setting_Train_Test_Load
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch
import matplotlib.pyplot as plt

if 1:
    # parameters
    np.random.seed(23)
    torch.manual_seed(23)

    # object initialization
    # data Loader for stage 4: Text Classification
    data_obj = Dataset_Loader('text_classification', 'movie review sentiment data')
    data_obj.dataset_source_folder_path = './data/stage_4_data/' 
    data_obj.dataset_file_name = 'text_classification'
    
    # Set text processing parameters
    data_obj.max_vocab_size = 15000
    data_obj.max_sequence_length = 200

    # use method RNN for Text Classification
    # Note: vocab_size will be dynamically updated by Setting_Train_Test_Load upon dataset loading
    method_obj = Method_RNN_Text('RNN-Text', 'vanilla recurrent neural network for text classification')

    # save results
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_4_result/RNN_Classification_'
    result_obj.result_destination_file_name = 'prediction_result'

    # setup pipeline settings manager
    setting_obj = Setting_Train_Test_Load('train test load', '')

    # evaluation metrics
    evaluate_obj = Evaluate_Metrics('binary classification metrics', '')

    # running section
    print('Starting Text Classification Task')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    
    # this handles loading, text normalization, dynamic vocabulary sizing, and mini-batch training
    metrics, loss_history = setting_obj.load_run_save_evaluate()

    # plotting section
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='green', label='Training Loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss Value')
    plt.title('RNN Training Convergence (Text Classification)')
    plt.legend()
    plt.grid(True)
    # Save to results folder
    plt.savefig('./result/stage_4_result/text_classification_learning_curve.png') 
    plt.close() # Safely close figure handle for headless executions

    print('Performance Metrics:')
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value}')
    print('Done')