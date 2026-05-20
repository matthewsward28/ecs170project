'''
Concrete IO class for text classification and text generation datasets
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import os
import re
from collections import Counter
import torch
import numpy as np

class Dataset_Loader(dataset):
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = ""
        self.dataset_file_name = dName if dName else ""
        
        # Hyperparameters for text tokenization and vocabulary clipping
        self.max_vocab_size = 15000
        self.max_sequence_length = 200
        self.vocab = {}

    def clean_text(self, text):
        '''Clean the text data by removing HTML tags, punctuation, and normalizing to lowercase'''
        text = text.lower()
        # Remove HTML line breaks common in movie reviews
        text = re.sub(r'<br\s*/?>', ' ', text)
        # Keep only lowercase letters and whitespace tokens
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()

    def build_vocabulary(self, train_texts):
        '''Build a unified word-to-index vocabulary dictionary based on training data token frequency'''
        word_counts = Counter()
        for tokens in train_texts:
            word_counts.update(tokens)
            
        # Extract the most common words based on max_vocab_size limit
        most_common = word_counts.most_common(self.max_vocab_size - 2)
        
        # Index 0 is reserved for padding, Index 1 is reserved for unknown words
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(most_common, start=2):
            self.vocab[word] = idx

    def text_to_sequence(self, tokens):
        '''Convert raw string token sequences into structured integers matching vocabulary indices'''
        sequence = []
        for token in tokens:
            if token in self.vocab:
                sequence.append(self.vocab[token])
            else:
                sequence.append(self.vocab['<UNK>'])
                
        # Truncate sequences that exceed maximum configured length threshold
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
        # Pad shorter sequences with trailing 0 values to balance multidimensional arrays
        else:
            sequence = sequence + [self.vocab['<PAD>']] * (self.max_sequence_length - len(sequence))
            
        return sequence

    def load_classification_data(self, base_path):
        '''Traverse directory folders to harvest train and test movie reviews'''
        data_splits = {}
        
        for split in ['train', 'test']:
            X_raw_tokens = []
            y_labels = []
            
            # Map subfolders directly to binary sequence evaluation classes
            for label_idx, sentiment in enumerate(['neg', 'pos']):
                folder_path = os.path.join(base_path, split, sentiment)
                print(f"Reading files from: {folder_path}")
                
                if os.path.exists(folder_path):
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith('.txt'):
                            file_path = os.path.join(folder_path, file_name)
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                raw_text = f.read()
                                tokens = self.clean_text(raw_text)
                                X_raw_tokens.append(tokens)
                                y_labels.append(label_idx)
                                
            data_splits[split] = {'tokens': X_raw_tokens, 'y': y_labels}
            
        return data_splits

    def load_generation_data(self, base_path):
        '''Read raw joke text lines from the CSV-formatted data file while stripping indices'''
        X_raw_tokens = []
        
        # Look for the jokes data file inside the specified root location path
        file = ['data']
        target_file = None
        
        for candidate in file:
            check_path = os.path.join(base_path, candidate)
            if os.path.exists(check_path):
                target_file = check_path
                break
                
        if target_file is None:
            if os.path.isfile(base_path):
                target_file = base_path
            else:
                for file_name in os.listdir(base_path):
                    if not file_name.endswith('.docx') and not file_name.startswith('.'):
                        target_file = os.path.join(base_path, file_name)
                        break

        if target_file is None:
            raise FileNotFoundError(f"Could not locate the text joke generation file inside: {base_path}")
            
        print(f"Reading generation elements from: {target_file}")
        with open(target_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            # Iterate through lines, skipping the header index row (Line 0: "ID","Joke")
            for line in lines[1:]:
                line = line.strip()
                if line:
                    # Strip the ID index number and leading comma (e.g., '1,"What did..."' -> '"What did..."')
                    # We look for the first comma split
                    parts = line.split(',', 1)
                    if len(parts) > 1:
                        joke_text = parts[1].strip().strip('"') # Remove wrapping quotation marks
                        tokens = self.clean_text(joke_text)
                        
                        if len(tokens) > 3: # Only collect jokes with functional text length
                            X_raw_tokens.append(tokens)
                        
        return X_raw_tokens
    
    def load(self):
        '''Main loader driver routing the data processing to text pipeline steps'''
        full_path = os.path.join(self.dataset_source_folder_path, self.dataset_file_name)
        print(f"Attempting to load dataset from path: {full_path}")
        
        # Text Classification
        if self.dataset_name == 'text_classification':
            raw_splits = self.load_classification_data(full_path)
            
            # Construct dictionary indices using exclusively training partition definitions
            self.build_vocabulary(raw_splits['train']['tokens'])
            print(f"Vocabulary successfully built. Total vocabulary size: {len(self.vocab)}")
            
            train_X = [self.text_to_sequence(t) for t in raw_splits['train']['tokens']]
            test_X = [self.text_to_sequence(t) for t in raw_splits['test']['tokens']]
            
            return {
                'train': {
                    'X': torch.LongTensor(np.array(train_X)), 
                    'y': torch.LongTensor(np.array(raw_splits['train']['y']))
                },
                'test': {
                    'X': torch.LongTensor(np.array(test_X)), 
                    'y': torch.LongTensor(np.array(raw_splits['test']['y']))
                }
            }
            
        # Text Generation
        elif self.dataset_name == 'text_generation':
            raw_jokes_tokens = self.load_generation_data(full_path)
            
            # Construct dictionary indices from all available jokes documents text strings
            self.build_vocabulary(raw_jokes_tokens)
            print(f"Vocabulary successfully built for generation. Total vocabulary size: {len(self.vocab)}")
            
            X_sequences = []
            y_sequences = []
            
            # Slide a sequence window over each joke to set up next-word target pairs
            # For each token index t: X contains tokens up to t, y contains the token at t+1
            for tokens in raw_jokes_tokens:
                # Convert whole joke line to integer tokens array matches
                numerical_tokens = [self.vocab[token] if token in self.vocab else self.vocab['<UNK>'] for token in tokens]
                
                for i in range(1, len(numerical_tokens)):
                    sub_seq = numerical_tokens[:i]
                    target_word = numerical_tokens[i]
                    
                    # Pad short history lists to match a uniform max_sequence_length configuration
                    if len(sub_seq) > self.max_sequence_length:
                        sub_seq = sub_seq[-self.max_sequence_length:]
                    else:
                        sub_seq = [self.vocab['<PAD>']] * (self.max_sequence_length - len(sub_seq)) + sub_seq
                        
                    X_sequences.append(sub_seq)
                    y_sequences.append(target_word)
                    
            print(f"Total sliced processing sequences generated: {len(X_sequences)}")
            
            # Since generation tasks utilize auto-regressive generation, we use the same pool as our testing benchmark
            return {
                'train': {
                    'X': torch.LongTensor(np.array(X_sequences)), 
                    'y': torch.LongTensor(np.array(y_sequences))
                },
                'test': {
                    'X': torch.LongTensor(np.array(X_sequences)), 
                    'y': torch.LongTensor(np.array(y_sequences))
                }
            }
            
        else:
            raise ValueError(f"Unknown dataset identity definition: {self.dataset_name}")