'''
This file process the training and testing procedure
'''
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import seaborn as sns
from torch.utils.data import DataLoader
from model import Euler_Maruyama_sampler, ScoreNet_embedding
from data import Loader


def dplot(df,mode,time,batch):
  """
  df: With Shape (points,axis)
  mode: "Pred" or "Thero"
  time: Int
  batch:Int
  """ 
  df=np.array(df)
  mean = np.mean(df, axis=0)
  cov = np.cov(df, rowvar=False)
  dist = multivariate_normal(mean=mean, cov=cov)

  # plt.scatter(df[:, 0], df[:, 1], s=10, c='blue', label='Data')
  # x, y = np.mgrid[mean[0]-3*np.sqrt(cov[0,0]):mean[0]+3*np.sqrt(cov[0,0]):.01, 
  #                 mean[1]-3*np.sqrt(cov[1,1]):mean[1]+3*np.sqrt(cov[1,1]):.01]
  x, y = np.mgrid[1:2:.01, 
              1:2:.01]
  pos = np.dstack((x, y))
  z = dist.pdf(pos)

  plt.contourf(x, y, z, levels=15, cmap='Reds', alpha=0.7)
  plt.legend()
  plt.xlabel("")
  plt.ylabel("")
  plt.savefig(f"PDF/{mode}-{time}-{batch}.png")
  plt.close()


device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

# Train Setting
batch_size =  5 #@param {'type':'integer'}
mode="Moment"

# Data Setting
datasize=5
N_points=40
diffusivity=0.01
time_series= np.arange(0, 1.01, 0.01)
shift = [0, 0]
data=Loader(datasize, N_points, diffusivity, time_series, shift)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

dplot(data.data.reshape(data.data.shape[1],-1,data.data.shape[3])[-1,:,:],"Theor",-1,0)  


# Model
score_model = ScoreNet_embedding(diff=diffusivity)
score_model = score_model.to(device)

# os.makedirs("ckpt/40/1e-3-MLP/",exist_ok=True)

# Inference
epochs=[40]
timehook=[1,10,50,70,99]
for epoch in epochs:
  path=f"PDF/40/lr3_Gau_e{epoch}/"
  ckpt = torch.load(f'ckpt/40/1e-3-spec/ckpt_{epoch}.pth', map_location=device)
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
  # samples=samples_1
  for i in timehook:
    samples=torch.cat([samples_1[len(samples_1)-i],samples_2[len(samples_2)-i],samples_3[len(samples_3)-i]],dim=1).cpu()
    # 计算Variance, Mean, Covariance
    varx=torch.var(torch.Tensor(data.data[:,i,:,0]),dim=1)  

    dplot(data.data.reshape(data.data.shape[1],-1,data.data.shape[3])[i,:,:],"Theor",i,0)  

    for j in range(len(data)):
      #Plot
      # dplot(data.data[j,i,:,:],"Theor",i,j)
      #
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

    dplot(samples.reshape(-1,samples.shape[2]),"Pred",i,0)  

    for j in range(len(data)):
        #Plot
        # dplot(samples[j,:,:],"Pred",i,j)
        #
        try:
            if abs(cov)<abs(torch.cov(samples[j,:,:].T)):
                cov=torch.cov(samples[j,:,:].T)
        except:
            cov=torch.cov(samples[j,:,:].T)
    meanx=torch.mean(samples[0,:,0])
    meany=torch.mean(samples[0,:,1])
    print(f"Pred----Epoch--{epoch}--Timestep-{i}: Varx--{torch.mean(varx)}; Vary--{torch.mean(vary)}; meanx--{meanx};meany--{meany}; Cov--{cov[0,1]}")
