'''
Main orchestration execution script for Stage 5 Graph Convolutional Network (GCN) tasks
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_5_code.Dataset_Loader import Dataset_Loader
from local_code.stage_5_code.Method_GCN import Method_GCN
from local_code.stage_5_code.Result_Saver import Result_Saver
from local_code.stage_5_code.Setting_Train_Test_Load import Setting_Train_Test_Load
from local_code.stage_5_code.Evaluate_Metrics import Evaluate_Metrics
import numpy as np
import torch
import matplotlib.pyplot as plt

if 1:
    # Set the target graph datasets to evaluate sequentially
    datasets_to_run = ['cora', 'citeseer', 'pubmed']
    
    # Configure separate plot lines colors for each network experiment
    plot_colors = {'cora': 'crimson', 'citeseer': 'purple', 'pubmed': 'darkgreen'}

    print('Stage 5 Graph Convolutional Network Pipeline')

    for current_dataset in datasets_to_run:
        print(f'\n>>> Executing Experiment for Graph: {current_dataset.upper()} <<<')
        
        # parameters parameters tracking seed anchors
        np.random.seed(42)
        torch.manual_seed(42)

        # Object initialization
        # Data Loader for stage 5 setup
        data_obj = Dataset_Loader(dName=current_dataset, dDescription=f'{current_dataset} citation network graph')
        data_obj.dataset_source_folder_path = './data/stage_5_data/' 
        data_obj.dataset_name = current_dataset

        # Configure layer dimensions matching specific dataset structures
        if current_dataset == 'cora':
            feature_dim = 1433
            class_dim = 7
        elif current_dataset == 'citeseer':
            feature_dim = 3703
            class_dim = 6
        elif current_dataset == 'pubmed':
            feature_dim = 500
            class_dim = 3

        # Use method GCN module structure setup
        method_obj = Method_GCN(
            mName=f'GCN-{current_dataset}', 
            mDescription='semi-supervised graph convolutional network',
            in_features=feature_dim,
            hidden_dim=16,
            num_classes=class_dim
        )

        # Save results structures independently to prevent overwriting across graph loops
        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = f'./result/stage_5_result/{current_dataset}_GCN_'
        result_obj.result_destination_file_name = 'node_predictions'

        # Setup pipeline settings manager
        setting_obj = Setting_Train_Test_Load('train test load', '')

        # Evaluation metrics structure
        evaluate_obj = Evaluate_Metrics('multi-class classification metrics', '')

        # Running section using the modular structural layout
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        
        # This handles parsing, message passing computations, full-graph epochs, and matrix saves
        # We catch the custom return payload structured inside our Method_GCN run method
        run_output = method_obj.run()
        
        pred_y = run_output['pred_y']
        true_y = run_output['true_y']
        loss_history = run_output['loss_history']

        # Evaluate metrics calculations over isolation test subsets
        evaluate_obj.data = {'true_y': true_y, 'pred_y': pred_y}
        metrics = evaluate_obj.evaluate()

        # Print performance metrics outputs to terminal console windows
        print(f"--- {current_dataset.upper()} Test Evaluation Metrics ---")
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value:.4f}')

        # Plotting section tracking individual learning convergence lines
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(loss_history) + 1), loss_history, color=plot_colors[current_dataset], linewidth=2, label='Training Loss')
        plt.xlabel('Training Epoch')
        plt.ylabel('Loss Value')
        plt.title(f'GCN Convergence Profile ({current_dataset.capitalize()} Dataset)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save independent learning curve diagrams to the results directory folder
        plot_save_path = f'./result/stage_5_result/gcn_{current_dataset}_learning_curve.png'
        plt.savefig(plot_save_path, dpi=150) 
        plt.close()
        print(f"Saved training learning curve figure to: {plot_save_path}")
        
    print('done')