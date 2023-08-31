"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """
    def __init__(self, input_dim, hidden_dim, dropout, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj, idx):

        eye = torch.eye(adj.shape[0]).to_sparse()
        adj_tilde = adj + eye
        z1 = self.fc1(torch.sparse.mm(adj_tilde, x_in))
        z1 = self.relu(z1)

        z1 = self.dropout(z1)

        x = self.fc2(torch.sparse.mm(adj_tilde, z1))
        ############## Task 10
    
        ##################
        # your code here #
        ##################
        
        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)
        
        ##################
        # your code here #
        ##################

        out = self.bn(out)
        zg = self.fc3(out)
        zg = self.dropout(zg)
        zg = self.relu(zg)
        out = self.fc4(zg)

        return F.log_softmax(out, dim=1)
