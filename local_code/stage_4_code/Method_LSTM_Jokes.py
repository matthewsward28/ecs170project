'''
Concrete MethodModule class for an auto-regressive gated sequential text generation MethodModule (LSTM)
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_LSTM_Jokes(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    # Text generation requires seeing sequences repeatedly; 20 epochs provides a clear look at optimization paths
    max_epoch = 20
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the LSTM model architecture,
    # how many layers, embedding size, recurrent hidden dimensions, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, vocab_size=5000, embedding_dim=128, hidden_dim=128):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # Store vocabulary size for dynamic reference during auto-regressive generation loops
        self.vocab_size = vocab_size
        
        # Word Embedding Layer: Maps tokenized index values to a dense continuous vector space
        # check here for nn.Embedding doc: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # Long Short-Term Memory Block: Introduces gated control vectors to maintain memory over sequences
        # batch_first=True expects matrix dimensions organized as [Batch, Sequence_Length, Embedding_Dim]
        # check here for nn.LSTM doc: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm_layer = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Final Fully Connected layers for next-token probability distribution predictions
        # Projects internal recurrent representations onto the full vocabulary size dimension space
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(hidden_dim, 128)
        self.activation_func_1 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        '''Forward propagation'''
        # Convert index integer tokens to dense feature coordinates
        # output shape: [Batch Size, Sequence Length, Embedding Dim]
        embedded = self.embedding_layer(x)
        
        # Propagate embeddings through sequential LSTM recurrent hidden and cell elements
        # lstm_out shape: [Batch Size, Sequence Length, Hidden Dim]
        # hn shape: [1, Batch Size, Hidden Dim] representing final temporal sequence hidden states
        # cn shape: [1, Batch Size, Hidden Dim] representing final temporal sequence cell states
        lstm_out, (hn, cn) = self.lstm_layer(embedded)
        
        # Isolate the final step hidden states to summarize information from entire sentence histories
        # Shape transforms from [1, Batch, Hidden] to [Batch, Hidden]
        h = hn.squeeze(0)
        
        # Fully connected projection layers
        h = self.activation_func_1(self.fc_layer_1(h))
        
        # output layer result
        # we return raw vocabulary logits, CrossEntropyLoss will handle normalized probability distributions
        y_pred = self.fc_layer_2(h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train_model(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()

        # It defines the size of the mini-batch (Maintained at 32 for memory protection)
        batch_size = 32
        # For the plot
        loss_history = []

        # Keep dataset arrays as flat, raw NumPy structures instead of giant Tensor object allocations.
        X_array = np.array(X)
        y_array = np.array(y)
        num_samples = X_array.shape[0]

        # it will be an iterative gradient updating process using mini-batches
        for epoch in range(self.max_epoch):
            # shuffle the data indices for each epoch to avoid bias
            indices = np.random.permutation(num_samples)
            X_shuffled = X_array[indices]
            y_shuffled = y_array[indices]
            
            epoch_loss = 0.0

            # mini-batch iteration
            for i in range(0, num_samples, batch_size):
                # Slice smaller partitions from raw arrays and construct Tensors dynamically on-the-fly
                X_batch = torch.LongTensor(X_shuffled[i : i + batch_size])
                y_batch = torch.LongTensor(y_shuffled[i : i + batch_size])

                # get the output through forward propagation
                y_pred = self.forward(X_batch)
                # calculate the training loss for the current batch
                train_loss = loss_function(y_pred, y_batch)

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # update the variables according to the optimizer and the gradients
                optimizer.step()

                # Explicitly isolate scalar float numbers via .item() to preserve memory overhead
                epoch_loss += train_loss.item()

            # Record the average loss of all batches for plotting
            avg_loss = epoch_loss / (num_samples / batch_size)
            loss_history.append(avg_loss)

            if epoch % 5 == 0:
                # Calculate accuracy on the last batch of the epoch for tracking
                pred_labels = y_pred.max(1)[1]
                acc = (pred_labels == y_batch).float().mean()
                print(f"Epoch {epoch} | Batch Acc: {acc.item():.4f} | Avg Loss: {avg_loss:.4f}")
            
        return loss_history
    
    def test(self, X):
        self.eval()
        
        # Loop over the test set in mini-batches to keep memory footprints low.
        X_array = np.array(X)
        num_samples = X_array.shape[0]
        batch_size = 64
        all_predictions = []
        
        # disable gradient calculation for efficiency during testing
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                X_batch = torch.LongTensor(X_array[i : i + batch_size])
                y_pred = self.forward(X_batch)
                batch_predictions = y_pred.max(1)[1].cpu().numpy()
                all_predictions.extend(batch_predictions)
                
        self.train() # set back to train mode after testing
        return np.array(all_predictions)

    def generate_joke(self, start_words, vocab_dict, max_generated_words=30, max_seq_length=200):
        '''Generate a complete text joke string starting from three provided seed words using LSTM logic'''
        self.eval()
        
        # Create an inverse lookup vocabulary map to convert integer IDs back to string tokens
        inverse_vocab = {idx: word for word, idx in vocab_dict.items()}
        
        # Convert starting words to vocabulary indices using safe dictionary access routes
        generated_indices = []
        for word in start_words:
            word = word.lower()
            generated_indices.append(vocab_dict.get(word, vocab_dict.get('<UNK>', 1)))
            
        # Run iterative loop to predict upcoming tokens auto-regressively
        with torch.no_grad():
            for _ in range(max_generated_words):
                # Isolate the current context tokens length window
                context = generated_indices[:]
                if len(context) > max_seq_length:
                    context = context[-max_seq_length:]
                else:
                    # Match the loader's design by pre-padding indices with leading 0 markers
                    context = [vocab_dict.get('<PAD>', 0)] * (max_seq_length - len(context)) + context
                    
                # Format to tensor batch structure: shape [1, max_seq_length]
                context_tensor = torch.LongTensor([context])
                
                # Forward propagate to retrieve vocabulary distribution logit values
                logits = self.forward(context_tensor)
                
                # Isolate index holding the absolute maximum score prediction probability
                predicted_idx = logits.max(1)[1].item()
                
                # Append predicted index token to keep building sequence context
                generated_indices.append(predicted_idx)
                
                # Break loop sequence early if padding or unknown tokens surface repeatedly
                if predicted_idx == vocab_dict.get('<PAD>', 0):
                    break
                    
        # Re-assemble integer collection into an readable output string piece
        output_words = [inverse_vocab.get(idx, '') for idx in generated_indices]
        self.train()
        return " ".join(output_words)
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}