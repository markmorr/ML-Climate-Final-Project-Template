# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 02:45:21 2022

@author: 16028
"""

#attempting to use gnn to better embed the complicated spatial relationships
# in us geographies
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

#using this for tutorial fucntions/to understand it
dataset = Planetoid(root='/tmp/Cora', name='Cora')
dataset = dataset.shuffle()
dataset = dataset.subsample(frac=0.5)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GINConv(nn=torch.nn.Sequential(torch.nn.Linear(2, 16),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(16, 16)))
        self.conv2 = GINConv(nn=torch.nn.Sequential(torch.nn.Linear(16, 16),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(16, 16)))
        self.conv3 = GINConv(nn=torch.nn.Sequential(torch.nn.Linear(16, 16),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(16, 16)))
        self.lin1 = torch.nn.Linear(16, 16)
        self.lin2 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out, data.y)
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
acc = correct / data.train_mask.sum().item()
print('Accuracy: {:.2f}'.format(acc))