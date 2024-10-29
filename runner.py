'''
This file process the training and testing procedure
'''
import os 
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import functools
from torch.optim import Adam
import torchvision.transforms as transforms
from model import ScoreNet, MLP, Euler_Maruyama_sampler, ScoreNet_embedding
from loss import APloss_fn
from data import Loader, pdfplot

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

# Train Setting
n_epochs =  40 #@param {'type':'integer'} 
batch_size =  5 #@param {'type':'integer'}
lr=0.001
mode="Moment"

# Data Setting
datasize=5
N_points=40
diffusivity=0.01
time_series= np.arange(0, 1.01, 0.01)
shift = [0, 0]
data=Loader(datasize, N_points, diffusivity, time_series, shift)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# Model
score_model = ScoreNet_embedding(diff=diffusivity)
# score_model.load_state_dict(torch.load("ckpt/40/1e-3-uniformflow/ckpt_99.pth"))
score_model = score_model.to(device)
optimizer = Adam(score_model.parameters(), lr=lr)

os.makedirs("ckpt/40/1e-3/",exist_ok=True)

# Training
for epoch in range(n_epochs):
  print(epoch)
  avg_loss = 0.
  num_items = 0
  for batch,(x,velocity) in enumerate(data_loader):
    x=torch.tensor(x,dtype=torch.float,device=device) # x with shape (batchsize, time, points, x-y)
    velocity=velocity.to(device)
    
    loss=0
    for i in range(1,x.shape[1]):
      optimizer.zero_grad()

      # every timestep
      time=(i+1e-5)/len(time_series) # normalize timestep to [0,1] 
      loss = APloss_fn(score_model, x[:,i,:,:], x[:,i-1,:,:], diffusivity, velocity[:,i-1,:,:,:], time, mode)

      loss.backward()    
      optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  print('Average Loss: {:5f}'.format(avg_loss / num_items))
  torch.save(score_model.state_dict(), f'ckpt/40/1e-3-spec/ckpt_{epoch}.pth')


# Inference
epochs=[20,40,60,80]
timehook=[1,10,20,50,80]
for epoch in epochs:
  ckpt = torch.load(f'ckpt/40/1e-3/ckpt_{epoch}.pth', map_location=device)
  score_model.load_state_dict(ckpt)
  sample_batch_size = 5
  sampler = Euler_Maruyama_sampler 

  ## Generate samples using the specified sampler.
  samples_1 = sampler(score_model,
                    torch.Tensor(data.data[:,-1,:,:]),  # init data
                    data.vel,
                    mode,
                    diffusivity,
                    sample_batch_size, 
                    device=device)
  samples_2 = sampler(score_model,
                    torch.Tensor(data.data[:,-1,:,:]),  # init data
                    data.vel,
                    mode,
                    diffusivity,
                    sample_batch_size, 
                    device=device)
  samples_3 = sampler(score_model,
                    torch.Tensor(data.data[:,-1,:,:]),  # init data
                    data.vel,
                    mode,
                    diffusivity,
                    sample_batch_size, 
                    device=device)

  for i in timehook:
    samples=torch.cat([samples_1[len(samples_1)-i],samples_2[len(samples_2)-i],samples_3[len(samples_3)-i]],dim=1).cpu()
    # 计算Variance, Mean, Covariance
    varx=torch.var(torch.Tensor(data.data[:,i,:,0]),dim=1)    
    for j in range(len(data)):
      try:
        if abs(cov)<abs(torch.cov(torch.Tensor(data.data[j,i,:,:].T))):
          cov=torch.cov(torch.Tensor(data.data[j,i,:,:].T))
      except:
        cov=torch.cov(torch.Tensor(data.data[j,i,:,:].T))
    meanx=torch.mean(torch.Tensor(data.data[0,i,:,0]))
    vary=torch.var(torch.Tensor(data.data[:,i,:,1]),dim=1)
    meany=torch.mean(torch.Tensor(data.data[0,i,:,1]))
    print(f"Theoretical----Epoch--{epoch}--Timestep-{i}: Varx--{torch.mean(varx)};Vary--{torch.mean(vary)};meanx--{meanx};meany--{meany}; Cov--{cov[0,1]}")

    varx=torch.var(samples[:,:,0],dim=1)
    vary=torch.var(samples[:,:,1],dim=1)
    for j in range(len(data)):
      try:
        if abs(cov)<abs(torch.cov(samples[j,:,:].T)):
            cov=torch.cov(samples[j,:,:].T)
      except:
            cov=torch.cov(samples[j,:,:].T)
    meanx=torch.mean(samples[0,:,0])
    meany=torch.mean(samples[0,:,1])
    print(f"Pred----Epoch--{epoch}--Timestep-{i}: Varx--{torch.mean(varx)}; Vary--{torch.mean(vary)}; meanx--{meanx};meany--{meany}; Cov--{cov[0,1]}")
      