#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 21:12:56 2021

check "proximity"" between database and points with wrong retreivals

alculate something like: w_j = exp( -sum_i([Tbm(i)-Tbd_j(i)]^2/si_i^2)) 
where i is channel index (over all channels we use), 
Tbm is the Tb of the measurement used, Tbd_j is the Tb of a database case j. 
si_i is the uncertainty of the channel. 
Just set to 1 for all channels to start with.
The cases with highest w are in some sense closest

@author: inderpreet
"""
import numpy as np
import xarray
import matplotlib.pyplot as plt
import pickle
from iwc2tb.GMI.gmiData import gmiData
import os

def get_args_closest_points_database(Tbd, Tbm):
    wj = np.zeros([Tbm.shape[0], Tbd.shape[0]])
    si = 1
#    for i in range(Tbm.shape[0]):
    for i in range(1000):
        print (i)
        for jx in range(Tbd.shape[0]):
            
            num = (Tbm[i, :] - Tbd[jx, :])**2
            
            wj[i, jx] = np.exp( -1.0 * np.sum(num/si**2)) 
            
    return np.argmax(wj, axis = 1)
        

def get_other_inputs(iargs, field):

    return field[iargs]     



#%%
#Tbd =  100 * np.random.rand(10000, 4) + 200
#Tbm =  100 * np.random.rand(100, 4) + 200
freq = ['166.5V', '166.5H', '183+-3', '183+-7']

with open("iargs.pickle", "rb") as f:
   iargs = pickle.load(f)
   f.close()    
   
#%%
batchSize          = 256
inputs             = np.array( ["ta", "t2m",  "wvp", "z0", "lat",  "stype"])
outputs            = "iwp"
xlog               = True
latlims            = [0, 45]
latlims            = [0, 65]   

#%% read training database
train    = gmiData(os.path.expanduser("~/Dendrite/Projects/IWP/GMI/training_data/TB_GMI_train_jan_lpa_pr1.nc"), 
                             inputs,
                             outputs,
                             batch_size = batchSize,
                             latlims = latlims,
                             normalise = None)
                             #log_iwp = xlog)


x = train.norm(train.x)

tiwp = train.y
tlat = train.lat
tlon = train.lon% 360
tlsm = train.stype

stype = np.argmax(tlsm, axis = 1)
stype = np.squeeze(stype)

tlsm = stype

mask = tlsm == 0
mask1 = train.x[:, 0] < 270

mask_train = np.logical_and(mask, mask1)
#mask_train  = mask1


#%% read cases with high IWP

test    = gmiData(os.path.expanduser("~/Dendrite/Projects/IWP/GMI/training_data/TB_GMI_train_jan_lpa.nc"), 
                             inputs,
                             outputs,
                             batch_size = batchSize,
                             latlims = latlims,
                             normalise = None)
                             #log_iwp = xlog)


xx = train.norm(test.x)

miwp = test.y.ravel()
mlat = test.lat.ravel()
mlon = test.lon% 360
mlsm = test.stype

stype = mlsm
stype = np.argmax(stype, axis = 1)
stype = np.squeeze(stype)

mlsm = stype

mask = stype == 0
mask1 = test.x[:, 0] < 225
mask2 = (test.x[:, 0] - test.x[:, 1] > 10) & (test.x[:, 0] - test.x[:, 1] < 20)


mask = np.logical_and(mask, mask1)

mask_test = np.logical_and(mask, mask2)



#%% extract closest database inputs 

iargs = get_args_closest_points_database(x[mask_train, :4], xx[mask_test, :4])

#%%
ctb  = get_other_inputs(iargs, train.x[mask_train, :4])
# cwvp = get_other_inputs(iargs, twvp)
clat = get_other_inputs(iargs, tlat[mask_train])
#clon = get_other_inputs(iargs, tlon)
# ct2m = get_other_inputs(iargs, tt2m)
clsm = get_other_inputs(iargs, tlsm[mask_train])
ciwp = get_other_inputs(iargs, tiwp[mask_train])
#%%
from matplotlib import cm
import matplotlib.colors as colors
fig, ax = plt.subplots(4, 1, figsize = [8, 20])
ax = ax.ravel()

y = np.arange(120, 250, 1)
yy = y

for ix in range(4):
    cs = ax[ix].scatter(test.x[mask_test, ix][:1000], ctb[:, ix][:1000], 
                   c = train.y.ravel()[iargs][:1000], s = 5, 
                   norm=colors.LogNorm(vmin=1e-4, vmax= 25), 
                  cmap = cm.gist_ncar)
    ax[ix].set_xlabel("measurements")
    ax[ix].set_ylabel("database")
    ax[ix].set_title(freq[ix])
    ax[ix].plot(y, yy, 'k')

fig.colorbar(cs, ax = ax)   
fig.savefig("proximity_tbs.pdf", bbox_inches = "tight")
#%%    

fig, ax = plt.subplots(2, 2, figsize = [10, 10])
ax = ax.ravel()

for ix, (field1, field2) in enumerate(zip([ mlat, miwp, mlsm], 
                              [clat, ciwp , clsm])):   
    ax[ix].scatter(field1[mask_test][:500], field2[:500])
    ax[ix].set_xlabel("measurements")
    ax[ix].set_ylabel("database")
    #ax[ix].set_title()

import pickle
with open("iargs.pickle", "wb") as f:
    pickle.dump(iargs, f)
    f.close()    
#%%    

fig, ax = plt.subplots(1, 1, figsize = [6, 6])
ax.scatter(miwp[mask_test][:1000], ciwp[:1000])
ax.set_xlabel("IWP ARO")
ax.set_ylabel("IWP TRO")

y = np.arange(0, 13, 1)
yy = y

ax.plot(y, yy, 'k')
fig.savefig("proximity_IWP.pdf", bbox_inches = "tight")

#%%

fig, ax = plt.subplots(1, 1, figsize = [6, 6])
ax.scatter(train.x[mask_train, 0][iargs][:1000], 
           train.x[mask_train, 0][iargs][:1000] - train.x[mask_train, 1][iargs][:1000], 
           label = "TRO")

ax.scatter(test.x[mask_test, 0][:1000], 
           test.x[mask_test, 0][:1000] - test.x[mask_test, 1][:1000], 
           label = "ARO")

ax.set_xlabel("TB 166 V [K]")
ax.set_ylabel("PD 166 V-H [K]")
ax.legend()

fig.savefig("proximity_PDs.pdf", bbox_inches = "tight")








   
            
            
        
        
        
        
        


