"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        ##################
        # your code here #
        ##################

        h0 = self.fc1(x_in)
        z0 = self.relu(torch.mm(adj, h0))
        z0 = self.dropout(z0)

        h1 = self.fc2(z0)
        z1 = self.relu(torch.mm(adj, h1))

        h2 = self.fc3(z1)
        x = self.dropout(h2)
        return F.log_softmax(x, dim=1), z1