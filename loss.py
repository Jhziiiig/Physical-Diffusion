'''
This file design the loss function
'''
import torch
import numpy as np


device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}


def marginal_prob_std(t, type,device=device):
  """return the std of p_{0t}(x(t) | x(0))
  Args:    
    t: A vector of time steps.0<t<1
    sigma: The $\sigma$ in our SDE.  
  
  Returns:
    std
  """    
  t = torch.tensor(t, device=device)
  if type==2:
    return torch.sqrt(0.03*t)
  else: 
    return torch.sqrt(0.2*t) 
 
def diffusion_coeff(t, type,device=device):
  """
  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    Diffusion coeff
  """
  # return torch.tensor(sigma**t, device=device)
  if type==2:
    return torch.tensor(np.sqrt(2*0.03), device=device) 
  else: 
    return torch.tensor(np.sqrt(0.2), device=device) 

def drift(t,type,data,device=device):
  """ return the time draft of SDE
  Args:    
    t: A vector of time steps.0<t<1
    sigma: The $\sigma$ in our SDE. 
    data: The data with all timestep  
  
  Returns:
    std
  """
  if type==0:
    return torch.tensor(0, device=device)
  elif type==1:
    return torch.tensor([t,0], device=device)
  elif type==2:
    drift=torch.tensor(1-(data[int(t*100),:,1])**2)[:,None]
    zero=torch.zeros(drift.shape)
    return torch.cat((drift,zero), dim=-1).to(device)


def loss_fn(model, x, marginal_prob_std,x_0, t):
  """The loss function for training score-based generative models.
  Args:
    model: U-Net
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  # print(x.shape) (1,100,2)
  loss=0
  std = marginal_prob_std(t)
  # z = ((x-mean_list[int(t*100)])/std)
  z = ((x-x_0)/std) 
  score = model(x, t) #(1,1000,2)
  loss = torch.mean(torch.sum((score * std + z)**2,dim=1),dim=1)
  # loss=loss[0,0]+loss[0,1]
  # print(loss.shape)
  return loss


def APloss_fn(model, x, x_ref, diff, drift, t, mode):
  """The loss function for training score-based generative models.
  Args:
    model: U-Net
    x: A mini-batch of training data.    
    diff: The diffusivity
    drift: The velocity
    x_ref: t-1
    x: t
  """
  # Interplote the drift function. drfit with shape (batchsize, x-vel, y-vel, x-y), 128 grids on x \in [0,pi]
  velx=[]
  vely=[]
  for i in range(x.shape[0]):
    velxb=[]
    velyb=[]
    for j in range(x.shape[1]):
      velxb.append(drift[i,int(x[i,j,0]/(np.pi/128)),int(x[i,j,1]/(np.pi/128)),0])
      velyb.append(drift[i,int(x[i,j,0]/(np.pi/128)),int(x[i,j,1]/(np.pi/128)),1])
    velx.append(velxb)
    vely.append(velyb) 
  vel=torch.cat([torch.Tensor(velx)[:,:,None],torch.Tensor(vely)[:,:,None]],dim=2).to(device) # [1,100,2]

  # vel=torch.cat([(1-x[:,:,1]**2)[:,:,None],(torch.zeros_like(x[:,:,0]))[:,:,None]],dim=-1).to(device)

  std = np.sqrt(diff) # number
  score = model(x, t, mode)
  x_exp=x-(vel-(std**2)*score)*0.01 + torch.sqrt(0.01) * std * torch.randn_like(x)

  loss=torch.mean(torch.sum(((x_ref-x_exp))**2,dim=1),dim=0)
  cov=0
  for i in range(x_ref.shape[0]):
    cov+=((torch.cov(x_ref[i].T)[0,1]-torch.cov(x_exp[i].T)[0,1])**2)/x_ref.shape[0]
  # loss=loss[0]+loss[1]+cov*5e5
  loss=loss[0]+loss[1]
  # loss=loss*(np.sqrt(0.01/t)/2)
  return loss