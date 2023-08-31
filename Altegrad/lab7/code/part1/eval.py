"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()
    y_true = list()
    for j in range(0, n_samples_per_card, batch_size):
        
        ############## Task 6
    
        ##################
        # your code here #
        ##################
        x_batch = X_test[i][j:j+batch_size]
        y_batch = y_test[i][j:j+batch_size]

        x_batch = torch.LongTensor(x_batch).to(device)
        y_batch = torch.FloatTensor(y_batch).to(device)

        output_deepsets = deepsets(x_batch)
        output_lstm = lstm(x_batch)
        
        y_pred_deepsets.append(output_deepsets)
        y_pred_lstm.append(output_lstm)
        y_true.append(y_batch)

    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()

    y_true = torch.cat(y_true)
    y_true = y_true.detach().cpu().numpy()
   
    acc_deepsets = accuracy_score(y_true.astype(int), y_pred_deepsets.astype(int))#your code here
    mae_deepsets = mean_absolute_error(y_true, y_pred_deepsets)#your code here
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    
    acc_lstm = accuracy_score(y_true.astype(int), y_pred_lstm.astype(int))#your code here
    mae_lstm = mean_absolute_error(y_true, y_pred_lstm)#your code here
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


############## Task 7
    
##################
# your code here #
##################

plt.figure()
plt.plot(cards, results['deepsets']['acc'], label='DeepSets accuracy')
plt.plot(cards, results['deepsets']['mae'], label='DeepSets MAE')
plt.legend()

plt.figure()
plt.plot(cards, results['lstm']['acc'], label='LSTM accuracy')
plt.legend()

plt.figure()
plt.plot(cards, results['lstm']['mae'], label='LSTM MAE')
plt.legend()

plt.show()