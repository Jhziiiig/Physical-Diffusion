import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader,Dataset,random_split


class Loader(Dataset):
    def __init__(self, datasize=25, N_points=100) -> None:
        """
        data["X"]: (2,1000,400)
        data["Up"]: (2,1000,400)
        data["Ub"]:  (2,1000,400)
        """
        super().__init__()
        self.X=[]
        self.Up=[]
        self.Ub=[]
        for i in range(datasize):
            data=sio.loadmat(f"/home/junhao/CVPR25/Velocity/dt0p01_T4_kappa0p002_1000_samples_FHIT_velo_1_source_{i+1}.mat")
            index=np.random.randint(1,1001,size=N_points)
            self.X.append(data["X"][:,index,:])
            self.Up.append(data["Up"][:,index,:])
            self.Ub.append(data["Ub"][:,index,:])
        self.X=np.array(self.X)
        self.Up=np.array(self.Up)
        self.Ub=np.array(self.Ub)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return np.transpose(self.X[index],(2,1,0)),np.transpose(self.Up[index],(2,1,0)), np.transpose(self.Ub[index],(2,1,0)) # (time, points, x-y)




