'''
This file design a normal U-Net NSCN
'''
import torch
import numpy as np
import torch.nn as nn
from torch.autograd.functional import jacobian


torch.manual_seed(42)
device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
N=3
K=[]
for i in range(1,N+1):
  for j in range(i+1):
    K.append([j,i-j])
K=[[1,1],[2,1],[1,2],[0,2],[2,0],[3,0],[0,3],[1,3],[3,1],[2,2],[0,4],[4,0]] # (14,2)  Remove Mean
K = nn.Parameter(torch.Tensor(K)[:,None,None,:].repeat(1,5,100,1), requires_grad=False) # (output_dim, batchsize, num_points, x-y)


## 采样步数
num_steps =  400 #@param {'type':'integer'}
def Euler_Maruyama_sampler(score_model, 
                           init_x,
                           u,
                           mode,
                           kappa, 
                           batch_size, 
                           timeseries, 
                           device='cuda', 
                           eps=1e-4):
  """Euler-Maruyama
  Args:
    score_model: PyTorch model.
    init_x: Initial X with shape [batchsize, N_points, x-y]
    drift: Velocity with shape [batchsize,Time,X,Y,x-y]
    batch_size
    num_steps
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps
  
  Returns:
    sample.    
  """
  step_size = torch.Tensor([timeseries[0] - timeseries[1]]).to(device)
  x=init_x.to(device)
  u=u.to(device)
  samples=[]
  with torch.no_grad():
    c=len(timeseries)
    for time_step in timeseries: 
      # Calculate the mean x
      g = float(np.sqrt(2*kappa))
      score=score_model(x, time_step, mode)
      x = x + ((g**2) * score - u[:,c-1,:,:]) * step_size + torch.sqrt(step_size) * g * torch.randn_like(x)
      samples.append(x)
      c=c-1
  return samples

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)



def invariant(x):
  x_mean=torch.mean(x,dim=1)[None, :, None, :] #[1, batchsize, 1, 2]
  x=x[None,:,:,:].repeat(12, 1, 1,1) # (32,5,40,2)
  x=x-x_mean.repeat(12, 1, x.shape[2],1) #[output_dim, batchsize, input_dim, 2]
  x=torch.pow(x,K) # [output_dim, batchsize, input_dim, 2]
  x=torch.sum(x[:,:,:,0]*x[:,:,:,1],dim=-1) # [output_dim, batchsize]
  x=x.permute(1,0)
  # x=torch.cat([x,x_mean[0,:,0,:]],dim=-1)
  return x


class InvarianceEncoding(nn.Module): 
  """
  Permutation Invariance Embedding
  """
  def __init__(self, batchsize=5, input_dim=40, output_dim=14, scale=1):
    super().__init__()
    self.od=output_dim
    self.bs=batchsize
    self.K = K

  def forward(self, x, mode):
    """
    Args:
      x:Input with shape [batchsize, input_dim, 2]  (5,40,2)
      mode: Fourier or Moment
    """
    x_mean=torch.mean(x,dim=1)[None, :, None, :] #[1, batchsize, 1, 2]
    x=x[None,:,:,:].repeat(self.od-2, 1, 1,1) # (32,5,40,2)
    x=x-x_mean.repeat(self.od-2, 1, x.shape[2],1) #[output_dim, batchsize, input_dim, 2]
    if mode=="Moment":
      x=torch.pow(x, self.K) # [output_dim, batchsize, input_dim, 2]
      x=torch.sum(x[:,:,:,0]*x[:,:,:,1],dim=-1) # [output_dim, batchsize] (32,5)
      x=x.permute(1,0)
      # x=torch.cat([x,x_mean[0,:,0,:]],dim=-1)
    elif mode=="Fourier":
      x=x*self.K  # [output_dim, batchsize, input_dim, 2]
      x=torch.sum(torch.cos(x[:,:,:,0]+x[:,:,:,1]),dim=-1) # [output_dim, batchsize]
      x=x.permute(1,0)
    return x

  def reverse(self, x):
    jac = jacobian(invariant, x) # [5, 32, 5, 40, 2]*[5,40]
    jac=torch.cat([jac[None,0,:,0,:,:],jac[None,1,:,1,:,:],jac[None,2,:,2,:,:],jac[None,3,:,3,:,:],jac[None,4,:,4,:,:]],dim=0)
    # print(jac.shape) # (5,32,40,2)
    return jac
  

class ScoreNet_embedding(nn.Module):
  """Unet"""
 
  def __init__(self, diff, i=12, h=64, o=128):
    """.
    Args:
      diff: The diffusivity.
      embed_dim: Same to 1.1-GaussianFourierProjection.
    """
    super().__init__()
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=h),
         nn.Linear(h, h))
    self.sym=InvarianceEncoding()
    self.diff = diff

    self.linear_model1 = nn.Sequential(
            nn.Linear(i, h),
            nn.Dropout(0.1),
            nn.GELU()
        )
        # Condition sigmas
        
    self.linear_model2 = nn.Sequential(
            nn.Linear(h, o),
            nn.Dropout(0.1),
            nn.GELU(),
            
            nn.Linear(o, i),
            nn.Dropout(0.1),
            nn.GELU(),
            
            nn.Linear(i, i),
    )

  def forward(self, x, t, mode):
    m=self.sym(x, mode)  #(5,32)
    h=self.linear_model1(m)
    h=self.linear_model2(h+self.embed(t).repeat(m.shape[0],1))/np.sqrt(self.diff*t)
    # h=self.linear_model2(h)/np.sqrt(self.diff*t)
    # print(h.shape) # (5,32)
    h=h[:,None,:]
    jac=self.sym.reverse(x) # (5,32,40,2)
    out=torch.zeros(jac.shape[0],jac.shape[2],jac.shape[-1]).to(device)
    for i in range(jac.shape[0]):
      for j in range(jac.shape[-1]):
        out[i,:,j]=torch.matmul(h[i,:,:],jac[i,:,:,j])[0,:]
    return out


class ScoreNet(nn.Module):
  """Unet"""
 
  def __init__(self, diff, i=32, h=64, o=128):
    """.
    Args:
      diff: The diffusivity.
      embed_dim: Same to 1.1-GaussianFourierProjection.
    """
    super().__init__()
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=h),
         nn.Linear(h, h))
    self.diff = diff

    self.linear_model1 = nn.Sequential(
            nn.Linear(2, h), 
            nn.Dropout(0.1),
            nn.GELU()
        )
        # Condition sigmas
        
    self.linear_model2 = nn.Sequential(
            nn.Linear(h, o),
            nn.Dropout(0.1),
            nn.GELU(),
            
            nn.Linear(o, i),
            nn.Dropout(0.1),
            nn.GELU(),
            
            nn.Linear(i, 2),
    )

  def forward(self, x, t, mode): 
    h=self.linear_model1(x)
    h=self.linear_model2(h+self.embed(t)[:,None,:].repeat(x.shape[0],x.shape[1],1))/np.sqrt(self.diff*t)
    return h
  

 
class MLP(nn.Module):
    def __init__(self, n_i=2, n_h=64, n_o=2, diff=0.01):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(n_i, n_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_o)
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=n_h),
                                    nn.Linear(n_h, n_h))
        self.sym=InvarianceEncoding()
        self.diff = diff
 
    def forward(self, input, t):
        return self.linear2(self.relu(self.linear1(input)+self.embed(t)))/np.sqrt(self.diff*t)
 

