from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Train_Test_Load import Setting_Train_Test_Load
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch
import matplotlib.pyplot as plt

if 1:
    # parameters
    np.random.seed(23)
    torch.manual_seed(23)

    # object initialization
    # data Loader for stage 3: CIFAR
    data_obj = Dataset_Loader('CIFAR', '')
    data_obj.dataset_source_folder_path = './data/stage_3_data/' 
    data_obj.dataset_file_name = 'CIFAR'
    
    # set dimensions for CIFAR-10 (32x32 RGB)
    data_obj.channel = 3
    data_obj.height = 32
    data_obj.width = 32

    # use method CNN for CIFAR
    method_obj = Method_CNN_CIFAR('CNN-CIFAR', 'convolutional neural network for cifar')

    # save results
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_3_result/CNN_CIFAR_'
    result_obj.result_destination_file_name = 'prediction_result'

    # data is presplit so we just load it
    setting_obj = Setting_Train_Test_Load('train test load', '')

    # evaluation metrics
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')

    # running section
    print('Starting CIFAR')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    
    # this now handles mini-batch training and returns loss history
    metrics, loss_history = setting_obj.load_run_save_evaluate()

    # plotting section
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='blue', label='Training Loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss Value')
    plt.title('CNN Training Convergence (CIFAR-10)')
    plt.legend()
    plt.grid(True)
    # Save to results folder
    plt.savefig('./result/stage_3_result/cifar_learning_curve.png') 
    plt.show()

    print('Performance Metrics:')
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value}')
    print('Done')