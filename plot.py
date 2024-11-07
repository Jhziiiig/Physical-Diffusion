'''
This file process the training and testing procedure
'''
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import seaborn as sns
from torch.utils.data import DataLoader
from model import Euler_Maruyama_sampler, ScoreNet_embedding
from data import Loader

colors=[(0.8,0.8,0.8),(0.3,0.6,0.9)]
n_bins=100
cmap_name="c"
cmap=LinearSegmentedColormap.from_list(cmap_name,colors)

def tabplot(array1, array2, array3, array4, time):
  fig, axs = plt.subplots(4, 1, figsize=(8, 10))  

  axs[0].plot(time, array1, color='r')  
  axs[0].set_title('Variance-X Error')  
  axs[0].set_ylabel(r"$1e3 \times \Delta Var_x$")
  axs[0].set_xlabel("Time")

  axs[1].plot(time, array2, color='g')
  axs[1].set_title('Variance-Y Error')
  axs[1].set_ylabel(r"$1e3 \times \Delta Var_y$")
  axs[1].set_xlabel("Time")

  axs[2].plot(time, array3, color='b')
  axs[2].set_title('Mean-X Error')
  axs[2].set_ylabel(r"$\Delta Mean_x$")
  axs[2].set_xlabel("Time")

  axs[3].plot(time, array4, color='y')
  axs[3].set_title('Mean-Y Error')
  axs[3].set_ylabel(r"$\Delta Mean_y$")
  axs[3].set_xlabel("Time")

  plt.tight_layout()
  plt.show()
   

def dplot(df,mode,time,batch,epoch):
  """
  df: With Shape (points,axis)
  mode: "Pred" or "Thero"
  time: Int
  batch:Int
  """ 
  df=np.array(df)
  x=df[:,0]
  y=df[:,1]

  # plt.scatter(df[:, 0], df[:, 1], s=10, c='blue', label='Data')
  # x, y = np.mgrid[mean[0]-3*np.sqrt(cov[0,0]):mean[0]+3*np.sqrt(cov[0,0]):.01, 
  #                 mean[1]-3*np.sqrt(cov[1,1]):mean[1]+3*np.sqrt(cov[1,1]):.01]
  # xx, yy = np.mgrid[-2.3:0.3:.05,-1.5:1.5:.05]
  xx, yy = np.mgrid[-0.5:1.5:.05,-0.8:0.8:.05]
  positions = np.vstack([xx.ravel(), yy.ravel()])
  values = np.vstack([x, y])
  kernel = gaussian_kde(values)
  f = np.reshape(kernel(positions).T, xx.shape)
  plt.contourf(xx, yy, f, levels=14, cmap="hot_r")

  # plt.colorbar()
  plt.legend()
  plt.xlabel("")
  plt.ylabel("")
  # plt.axis("equal")
  plt.savefig(f"PDF/100/lr3_Poi_e{epoch}/{mode}-{time}-{batch}.png")
  # plt.show()
  plt.close()

def gauplot(df,mode,time,batch,epoch):
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

  x, y = np.mgrid[1:2:.01, 
              1:2:.01]
  pos = np.dstack((x, y))
  z = dist.pdf(pos)

  fig, ax = plt.subplots()
  contourf = ax.contourf(x, y, z, levels=15, cmap='hot_r', alpha=0.7)
  plt.legend()
  plt.xlabel("")
  plt.ylabel("")
  # ax.set_facecolor("grey")
  ax.set_aspect('equal')
  plt.savefig(f"PDF/100/lr3_Gau_e{epoch}/{mode}-{time}-{batch}.png")
  # plt.show()
  plt.close()


device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

# Train Setting
batch_size =  5 #@param {'type':'integer'}
mode="Moment"
np.random.seed(60)  # fixed 60


# Data Setting
datasize=5
N_points=100
diffusivity=0.01
time_series= np.arange(0, 1.01, 0.01)
# shift = [-np.pi / 2, -np.pi / 2]
shift = [0,0]
data=Loader(datasize, N_points, diffusivity, time_series, shift)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

# dplot(data.data.reshape(data.data.shape[1],-1,data.data.shape[3])[-50,:,:],"Theor",-1,0)  
# for i in [1,10,20,30,40,50,60,70,80,90,99]:
#   dplot(data.data[0,i,:,:],"Theor",i,0,40)

# Model
score_model = ScoreNet_embedding(diff=diffusivity)
score_model = score_model.to(device)

# Inference
epochs=[40]
# timehook=[1,10,50,70,99]
timehook=[i+1 for i in range(98)]
for epoch in epochs:
  varx_l=[]
  vary_l=[]
  meanx_l=[]
  meany_l=[]
  
  path=f"PDF/100/lr3_Gau_e{epoch}/"
  os.makedirs(path,exist_ok=True)
  # ckpt = torch.load(f'ckpt/100/1e-3-Poi/ckpt_{epoch}.pth', map_location=device)
  ckpt = torch.load('/home/junhao/CVPR25/ckpt/Gaussian-lr1e-3-epoch20.pth',map_location=device)
  score_model.load_state_dict(ckpt)
  sample_batch_size = 5
  sampler = Euler_Maruyama_sampler 

  ## Generate samples using the specified sampler.
  samples = sampler(score_model,
                    torch.Tensor(data.data[:,-1,:,:]),  # init data
                    data.vel,
                    mode,
                    diffusivity,
                    sample_batch_size, 
                    device=device)
  # samples=samples_1
  for i in timehook:
    sample=samples[-i][0,:,:].cpu()
    # 计算Variance, Mean, Covariance
    varx=torch.var(torch.Tensor(data.data[0,i,:,0]),dim=0)  
    # gauplot(data.data[0,i,:,:],"Theor",i,0,epoch)  

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
    vary=torch.var(torch.Tensor(data.data[0,i,:,1]),dim=0)
    meany=torch.mean(torch.Tensor(data.data[0,i,:,1]))
    print(f"Theoretical----Epoch--{epoch}--Timestep-{i}: Varx--{torch.mean(varx)};Vary--{torch.mean(vary)};meanx--{meanx};meany--{meany}; Cov--{cov[0,1]}")

    varx_p=torch.var(sample[:,0],dim=0)
    vary_p=torch.var(sample[:,1],dim=0)

    gauplot(sample,"Pred",i,0,epoch)  

    for j in range(1):
        #Plot
        # dplot(samples[j,:,:],"Pred",i,j)
        #
        try:
            if abs(cov)<abs(torch.cov(sample[:,:].T)):
                cov_p=torch.cov(sample[:,:].T)
        except:
            cov_p=torch.cov(sample[:,:].T)
    meanx_p=torch.mean(sample[:,0])
    meany_p=torch.mean(sample[:,1])

    varx_l.append((varx_p-varx)*1000)
    vary_l.append((vary_p-vary)*1000)
    meanx_l.append((meanx-meanx_p))
    meany_l.append((meany-meany_p))
    print(meany)
    print(meany_p)
    print(f"Pred----Epoch--{epoch}--Timestep-{i}: Varx--{torch.mean(varx_p)}; Vary--{torch.mean(vary_p)}; meanx--{meanx_p};meany--{meany_p}; Cov--{cov_p[0,1]}")

  tabplot(varx_l, vary_l, meanx_l, meany_l, np.array(timehook)/100)


