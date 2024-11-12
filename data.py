import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader,Dataset,random_split


def sample_points_2DHIT(N_points, diffusivity, time_series, shift, velocity_index, case):
    # Be mindful that your time_series must be dense enough, otherwise the
    # simulation could blow up and give weird results

    # The size of locations is (2, N_points, len(time_series))
    starting_location = np.array([np.pi / 2, np.pi / 2]) + np.array(shift).flatten()
    L = np.pi
    Deltat = 0.01
    xx = np.linspace(0, L, 129)
    dx = xx[1] - xx[0]
    xx = xx + dx / 2
    # Recommended value for diffusivity = 1/500;
    # initialization
    data = sio.loadmat(f"/home/junhao/CVPR25/Velocity/recordinguv{velocity_index}.mat")
    U_rec = data['U_rec']
    V_rec = data['V_rec']

    vel=[]
    vel=np.vstack([U_rec[None,:],V_rec[None,:]])

    if case=="Gau" or case=="Poi":
        vel=np.zeros_like(vel)
        # print(vel.shape)

    locations = np.repeat(starting_location[:, np.newaxis], N_points, axis=1)
    locations = np.expand_dims(locations, axis=2)

    for i in range(1, len(time_series)):
        dt = time_series[i] - time_series[i - 1]
        ctl = int(np.floor(time_series[i] / Deltat))
        alpha = (time_series[i] % Deltat) / Deltat
        U_cur = (1 - alpha) * U_rec[:, :, ctl] + alpha * U_rec[:, :, ctl + 1]
        V_cur = (1 - alpha) * V_rec[:, :, ctl] + alpha * V_rec[:, :, ctl + 1]
        U_cur = np.vstack([U_cur, U_cur[0, :]])
        U_cur = np.hstack([U_cur, U_cur[:, 0][:, np.newaxis]])
        V_cur = np.vstack([V_cur, V_cur[0, :]])
        V_cur = np.hstack([V_cur, V_cur[:, 0][:, np.newaxis]])

        # Create interpolants for U and V
        FU = RegularGridInterpolator((xx, xx), U_cur.T, method='linear')
        FV = RegularGridInterpolator((xx, xx), V_cur.T, method='linear')

        # Calculate velocities using interpolation
        loc_x = np.mod(locations[0, :, -1] - dx / 2, L) + dx / 2
        loc_y = np.mod(locations[1, :, -1] - dx / 2, L) + dx / 2
        velocity_x = FU((loc_x, loc_y))
        velocity_y = FV((loc_x, loc_y))

        velocity = np.array([velocity_x, velocity_y])

        if case=="Vel":
            new_locations = locations[:, :, -1] + velocity * dt + np.sqrt(2 * diffusivity) * np.random.randn(*locations[:, :, -1].shape) * np.sqrt(dt)
        elif case=="Gau":
            new_locations = locations[:, :, -1] + np.sqrt(2 * diffusivity) * np.random.randn(*locations[:, :, -1].shape) * np.sqrt(dt)
        elif case=="Poi":
            new_locations = locations[:, :, -1] + np.sqrt(2 * diffusivity) * np.random.randn(*locations[:, :, -1].shape) * np.sqrt(dt)
            new_locations[0,:]=new_locations[0,:]+dt*(1-locations[1,:,-1]**2)

        locations = np.concatenate((locations, new_locations[:, :, np.newaxis]), axis=2)

    return locations.transpose(2,1,0),np.array(vel).transpose(3,1,2,0)



class Loader(Dataset):
    def __init__(self, datasize=25, N_points=100,diffusivity=0.01,time_series= np.arange(0, 1.01, 0.01),shift = [0, 0], case="Gau") -> None:
        """ The shape of data is (datasize, time, points,x-y), the shape of vel is (datasize,time,velocity-x,velocity-y,x-y)
        The data consists of different velocity
        """
        super().__init__()
        index=np.random.choice(np.arange(100),datasize,replace=False)+1
        test=[i+1 for i in range(100) if i+1 not in index]
        for i in index:
            if i!=11:
                try:
                    point,spe=sample_points_2DHIT(N_points, diffusivity, time_series, shift, i, case)
                    data=np.vstack([data,point[None,:]])
                    vel=np.vstack([vel,spe[None,:]])
                except UnboundLocalError:
                    data,vel=sample_points_2DHIT(N_points, diffusivity, time_series, shift, i, case)
                    data=data[None,:]
                    vel=vel[None,:]
        for i in test:
            if i!=11:
                try:
                    tpoint,tspe=sample_points_2DHIT(N_points, diffusivity, time_series, shift, i, case)
                    tdata=np.vstack([tdata,tpoint[None,:]])
                    tvel=np.vstack([tvel,tspe[None,:]])
                except UnboundLocalError:
                    tdata,tvel=sample_points_2DHIT(N_points, diffusivity, time_series, shift, i, case)
                    tdata=tdata[None,:]
                    tvel=tvel[None,:]

        self.data=data
        self.vel=vel
        self.tdata=tdata
        self.tvel=tvel
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.vel[index]




