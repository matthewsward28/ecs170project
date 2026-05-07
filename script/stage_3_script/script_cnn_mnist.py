from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
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
    # data Loader for stage 3: MNIST
    data_obj = Dataset_Loader('MNIST', 'hand-written digit recognition')
    data_obj.dataset_source_folder_path = './data/stage_3_data/MNIST/' 
    data_obj.train_file_name = 'train.csv' 
    data_obj.test_file_name = 'test.csv'
    
    # set dimensions for MNIST (28x28 grayscale)
    data_obj.channel = 1
    data_obj.height = 28
    data_obj.width = 28

    # use method CNN for MNIST
    method_obj = Method_CNN_MNIST('CNN-MNIST', 'convolutional neural network for mnist')

    # save results
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_3_result/CNN_MNIST_'
    result_obj.result_destination_file_name = 'prediction_result'

    # data is presplit so we just load it
    setting_obj = Setting_Train_Test_Load('train test load', '')

    # evaluation metrics
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')

    # running section
    print('Starting MNIST')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    
    # this call performs training with mini-batches and returns loss history
    metrics, loss_history = setting_obj.load_run_save_evaluate()

    # plotting section
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='blue', label='Training Loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss Value')
    plt.title('CNN Training Convergence (MNIST)')
    plt.legend()
    plt.grid(True)
    # save to results folder
    plt.savefig('./result/stage_3_result/mnist_learning_curve.png') 
    plt.show()

    print('Performance Metrics:')
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value}')
    print('Done')