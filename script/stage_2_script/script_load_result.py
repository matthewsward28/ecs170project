import pickle

file_path = './result/stage_2_result/MLP_prediction_result'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print("Keys in the file:", data.keys())
print("First 10 True Labels:", data['true_y'][:10])
print("First 10 Predictions:", data['pred_y'][:10])