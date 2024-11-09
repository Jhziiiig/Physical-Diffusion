'''
This file design the loss function
'''
import torch
import numpy as np


device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}

noise_term = np.random.normal(0, 1, 20)
noise_term=torch.Tensor(noise_term).to(device)

def APloss_fn(model, x, kappa, up, t, mode, timestep):
  """
  Implement partially known velocity function.
  Args:
    x:(batchsize, n_points, x-y)
    up:(batchsize, x-vel, y-vel, x-y)
  """

  std = np.sqrt(2*kappa) # number
  noise=torch.sqrt(torch.Tensor([timestep]).to(device)) * std * noise_term
  J=0
  for i in range(len(noise_term)):
    score = model(x+noise[i], t, mode)
    J+=(std**2*score*timestep-noise[i])**2
  J=J/len(noise_term)
  loss=torch.mean(torch.sum(J,dim=1),dim=0)
  loss=loss[0]+loss[1]
  return loss
