"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sparse_mx_to_torch_sparse_tensor

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 6
    
        ##################
        # your code here #
        ##################

        eye = torch.eye(adj.shape[0]).to_sparse()
        adj_tilde = adj + eye
        z1 = self.fc1(torch.sparse.mm(adj_tilde, x_in))
        z1 = self.relu(z1)

        x = self.fc2(torch.sparse.mm(adj_tilde, z1))
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros((int(torch.max(idx).item())+1,x.size(1))).to(self.device)
        out = out.scatter_add_(0, idx, x) 
        
        ##################
        # your code here #
        ##################

        zg = self.fc3(out)
        zg = self.relu(zg)
        out = self.fc4(zg)

        return F.log_softmax(out, dim=1)
