from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Load import Setting_Train_Test_Load
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch
import matplotlib.pyplot as plt

if 1:
    # parameters
    np.random.seed(23)
    torch.manual_seed(23)

    # object initialization
    # Data Loader for stage 2
    data_obj = Dataset_Loader('stage 2 data', '')
    data_obj.dataset_source_folder_path = './data/stage_2_data/' 
    data_obj.train_file_name = 'train.csv' 
    data_obj.test_file_name = 'test.csv'

    # Use method MLP
    method_obj = Method_MLP('multi-layer perceptron', '')

    # Save results
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    # Data is presplit so we just load it
    setting_obj = Setting_Train_Test_Load('train test load', '')

    # Evaluation metrics
    evaluate_obj = Evaluate_Metrics('multiclass metrics', '')

    # Running section
    print('************ Start Stage 2 ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)

    # Plotting section
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='blue', label='Training Loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss Value')
    plt.title('MLP Training Convergence (Stage 2)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./result/stage_2_result/learning_curve.png') # Save to results folder
    plt.show()
    
    # Returns a dictionary of metrics
    metrics = setting_obj.load_run_save_evaluate()
    
    print('************ Overall Performance ************')
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value}')
    print('************ Finish ************')