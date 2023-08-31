import torch
import torch.nn as nn

x = torch.ones(5)
x[1]=2
x[2]=2
print(x)
emb = nn.Embedding(5, 3)
x = emb(x.long())
print(x, torch.sum(x, axis=1))
