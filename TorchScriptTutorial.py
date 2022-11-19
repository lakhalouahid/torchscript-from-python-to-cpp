#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn


device = torch.device("cpu")



## Define the synthetic data

x = torch.linspace(-0.25, 0.75, 100)
noise = torch.randn(100) / 20
noised_x =  x + noise
yn = 2 * noised_x**2 - noised_x + 3

## Define neural network
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x
    

model = Network()
model.to(device)


## Miscellaneous settings
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_func = nn.MSELoss(reduction="sum")



## Training the neural net

x = x[..., None].to(device)
yn = yn[..., None].to(device)
train = False
if train:
  model.train()
  while True:
      ypred = model(x)
      loss = loss_func(ypred, yn)
      optimizer.zero_grad()
      loss.backward()
      print(loss)
      optimizer.step()
      if loss.item() < 2.5e-1:
          break
  torch.save(model.state_dict(), r'train.pkl')

## Save model weights
model.load_state_dict(torch.load(r'train.pkl'))



## Convert model to graph
script = torch.jit.trace(model, (x)) # for static graphs
script = torch.jit.script(model) # for dynamic graphs
script.save("model_graph.pt") # save graph to disk so we can access this graph from libtorch cpp API



## Python host output
model.eval()
x = torch.linspace(-0.25, 0.75, 10).reshape(-1, 1)
y = model(x)
print("input:", x)
print("output:", y)
