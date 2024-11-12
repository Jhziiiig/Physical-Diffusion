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
from data import Loader
from plot import tabplot, gauplot, dplot

import time

start_time = time.time()

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
np.random.seed(42)
torch.manual_seed(0)

# Train Setting
n_epochs =  0
batch_size =  5 
lr=1e-3
mode="Moment"

# Data Setting
case="Vel"
datasize=70 # 70
N_points=100 # 100
kapa=0.001
time_series= np.arange(0, 1.01, 0.01)
delta_t=torch.tensor([time_series[1]-time_series[0]]).to(device)
shift = [0,0]
data=Loader(datasize, N_points, kapa, time_series, shift, case)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
# Model
score_model = ScoreNet_embedding(diff=kapa)
# score_model = ScoreNet(diff=kapa)

score_model = score_model.to(device)
optimizer = Adam(score_model.parameters(), lr=lr)

os.makedirs("ckpt/100/1e-3-Vel",exist_ok=True)

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

      time=(i+1e-5)/len(time_series) # normalize timestep to [0,1] 
      loss = APloss_fn(score_model, x[:,i,:,:], x[:,i-1,:,:], kapa, velocity[:,i-1,:,:,:], time, mode, delta_t)
      loss.backward()    
      optimizer.step()
      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]
  print('Average Loss: {:9f}'.format(avg_loss / x.shape[1]*num_items))
  torch.save(score_model.state_dict(), f'ckpt/100/1e-3-Vel/ckpt_{epoch+1}.pth')


end_time = time.time()
print(f"Time consuming:{end_time - start_time} seconds")

np.random.seed(60)  # fixed 60

# Model
# score_model = ScoreNet_embedding(diff=kapa)
# score_model = score_model.to(device)
sampler = Euler_Maruyama_sampler 
sample_batch_size = 5

# Inference
epochs=[30,40,50]
# timehook=[1,10,50,70,99]
timehook=[i+1 for i in range(99)]
for epoch in epochs:
  varx_l=[]
  vary_l=[]
  meanx_l=[]
  meany_l=[]
  
  path=f"PDF/100/lr3_Vel_e{epoch}/"
  os.makedirs(path,exist_ok=True)
  # ckpt = torch.load(f'ckpt/100/1e-3-Vel/ckpt_{epoch}.pth', map_location=device)
  ckpt=torch.load(F"/home/junhao/CVPR25/ckpt/Iso_lr1e-3_30epoch_Moment.pth", map_location=device)
  score_model.load_state_dict(ckpt)

  ## Generate samples using the specified sampler.
  try:
    samples = sampler(score_model,
                      torch.Tensor(data.tdata[:,-1,:,:]),  # init data
                      data.tvel,
                      mode,
                      kapa,
                      case,
                      sample_batch_size, 
                      device=device)

    for i in timehook:
      cov_l=[]
      cov_pl=[]
      varx_mid=[]
      vary_mid=[]
      meanx_mid=[]
      meany_mid=[]
      for j in range(len(data.tdata)):
          try:
            if abs(cov)<abs(torch.cov(torch.Tensor(data.tdata[j,i,:,:].T))):
              cov=torch.cov(torch.Tensor(data.tdata[j,i,:,:].T))
          except:
            cov=torch.cov(torch.Tensor(data.tdata[j,i,:,:].T))
          cov_l.append(cov)

      for j in range(len(samples[0])):
          try:
              if abs(cov)<abs(torch.cov(samples[-i][j,:,:].T)):
                  cov_p=torch.cov(samples[-i][j,:,:].T)
          except:
              cov_p=torch.cov(samples[-i][j,:,:].T)
          cov_pl.append(cov_p)

      for k in range(len(data.tdata)):
        # 计算Variance, Mean, Covariance
        dplot(data.tdata[k,i,:,:],"Theor",i,k,epoch)  
        varx=torch.var(torch.Tensor(data.tdata[k,i,:,0]),dim=0)  
        meanx=torch.mean(torch.Tensor(data.tdata[k,i,:,0]))
        vary=torch.var(torch.Tensor(data.tdata[k,i,:,1]),dim=0)
        meany=torch.mean(torch.Tensor(data.tdata[k,i,:,1]))

        sample=samples[-i][k,:,:].cpu()
        dplot(sample,"Pred",i,k,epoch)  
        varx_p=torch.var(sample[:,0],dim=0)
        vary_p=torch.var(sample[:,1],dim=0)
        meanx_p=torch.mean(sample[:,0])
        meany_p=torch.mean(sample[:,1])

        varx_mid.append((varx_p-varx)*1000)
        vary_mid.append((vary_p-vary)*1000)
        meanx_mid.append((meanx-meanx_p))
        meany_mid.append((meany-meany_p))

        with open("result.txt", "a") as f:
          f.write(f"Theoretical----Epoch--{epoch}--Timestep-{i}: Varx--{torch.mean(varx)};Vary--{torch.mean(vary)};meanx--{meanx};meany--{meany}.\n")
          f.write(f"Pred----Epoch--{epoch}--Timestep-{i}: Varx--{torch.mean(varx_p)}; Vary--{torch.mean(vary_p)}; meanx--{meanx_p};meany--{meany_p}.\n")
      varx_l.append(varx_mid)
      meanx_l.append(meanx_mid)
      vary_l.append(vary_mid)
      meany_l.append(meany_mid)
    tabplot(np.array(varx_l), np.array(vary_l), np.array(meanx_l), np.array(meany_l), np.array(timehook)/100)
  except ValueError or OverflowError:
    print(epoch)
    pass


