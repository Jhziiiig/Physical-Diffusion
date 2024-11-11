'''
This file process the training and testing procedure
'''
import os 
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import ScoreNet_embedding
from loss import APloss_fn
from data import Loader
import matplotlib.pyplot as plt

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
np.random.seed(42)
torch.manual_seed(0)

# Train Setting
n_epochs =  100 #@param {'type':'integer'} 
batch_size =  5 #@param {'type':'integer'}
lr=1e-3
mode="Moment"

# Data Setting
datasize=10
N_points=100
kapa=0.2
time_series= np.arange(0, 1.00, 0.0025)
step_size=np.abs(time_series[0]-time_series[1])
data=Loader(datasize, N_points)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Model
score_model = ScoreNet_embedding(diff=kapa)
# score_model.load_state_dict(torch.load("ckpt/40/1e-3-uniformflow/ckpt_99.pth"))
score_model = score_model.to(device)
optimizer = Adam(score_model.parameters(), lr=lr)
losslist=[]

os.makedirs("ckpt/100/1e-3-Vel",exist_ok=True)

# Training
for epoch in range(n_epochs):
  print(epoch)
  avg_loss = 0.
  num_items = 0
  for batch,(x, up, ub) in enumerate(data_loader):
    x=torch.tensor(x,dtype=torch.float,device=device) # x with shape (batchsize, time, points, x-y)
    up=up.to(device)
    ub=ub.to(device)   
    loss=0
    for i in range(x.shape[1]):
      optimizer.zero_grad()
      # every timestep
      time=(i+1e-5)/len(time_series) # normalize timestep to [0,1] 
      loss = APloss_fn(score_model, x[:,i,:,:], kapa, up[:,i,:,:], time, mode, timestep=step_size)
      # print(loss)
      loss.backward()    
      optimizer.step()
      avg_loss += loss.item()/x.shape[1]
    num_items += x.shape[0]
  print('Average Loss: {:12f}'.format(avg_loss / num_items))
  torch.save(score_model.state_dict(), f'ckpt/100/1e-3-Vel/ckpt_{epoch+1}.pth')
  losslist.append(avg_loss / num_items)
plt.plot(losslist)
plt.savefig(f"{lr}-loss.png")
