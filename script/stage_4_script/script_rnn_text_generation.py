'''
Main orchestration execution script for Stage 4 text generation (Jokes) tasks
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Jokes import Method_RNN_Jokes
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
    # data Loader for stage 4: Text Generation (Jokes CSV directory path)
    data_obj = Dataset_Loader('text_generation', 'short jokes text dataset')
    data_obj.dataset_source_folder_path = './data/stage_4_data/' 
    data_obj.dataset_file_name = 'text_generation'
    
    # Set sequential modeling hyperparameters
    data_obj.max_vocab_size = 8000
    data_obj.max_sequence_length = 30 # Jokes are short, keeping windows tight improves memory on CPU

    # use method RNN for Text Generation
    # Note: vocab_size will be dynamically updated by Setting_Train_Test_Load upon dataset loading
    method_obj = Method_RNN_Jokes('RNN-Jokes', 'vanilla recurrent neural network for word generation')

    # save results structure
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_4_result/RNN_Generation_'
    result_obj.result_destination_file_name = 'generation_result'

    # setup pipeline settings manager
    setting_obj = Setting_Train_Test_Load('train test load', '')

    # evaluation metrics placeholder
    evaluate_obj = Evaluate_Metrics('generation metrics placeholder', '')

    # running section
    print('Starting Text Generation Task')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    
    # this handles loading, text normalization, sliding-window tokenization, and training loops
    metrics, loss_history = setting_obj.load_run_save_evaluate()

    # plotting section
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='purple', label='Training Loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss Value')
    plt.title('RNN Training Convergence (Joke Generation)')
    plt.legend()
    plt.grid(True)
    # Save learning curves to the results folder directory
    plt.savefig('./result/stage_4_result/text_generation_learning_curve.png') 
    plt.close()

    # Grab the vocab map built during loading steps to feed token lookups
    vocabulary_mapping = data_obj.vocab

    # Trial 1: Prompt assignment requested check using words present in the training set
    seed_words_1 = ["what", "did", "the"]
    print(f"Input Seed Words: {seed_words_1}")
    generated_story_1 = method_obj.generate_joke(seed_words_1, vocabulary_mapping, max_generated_words=25)
    print(f"Generated Joke Output:\n-> \"{generated_story_1}\"\n")

    # Trial 2: Prompt assignment requested check using completely random words out of your head
    seed_words_2 = ["knock", "knock", "who"]
    print(f"Input Seed Words: {seed_words_2}")
    generated_story_2 = method_obj.generate_joke(seed_words_2, vocabulary_mapping, max_generated_words=25)
    print(f"Generated Joke Output:\n-> \"{generated_story_2}\"")
    
    print('\nDone')