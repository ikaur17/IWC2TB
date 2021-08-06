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

def get_args_closest_points_database(Tbd, Tbm):
    wj = np.zeros([Tbm.shape[0], Tbd.shape[0]])
    si = 1
    for i in range(Tbm.shape[0]):
        print (i)
        for jx in range(Tbd.shape[0]):
            
            num = (Tbm[i, :] - Tbd[jx, :])**2
            
            wj[i, jx] = np.exp( -1.0 * np.sum( 0.5 * num/si**2)) 
            
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
    

#%% read training database
train = xarray.open_dataset("/home/inderpreet/Dendrite/Projects/IWP/GMI/training_data/TB_GMI_train.nc") 

tiwp = train.ta.iwp
tlat = train.ta.lat
tlon = train.ta.lon% 360
tbd  = train.ta
tt2m = train.ta.t2m
twvp = train.ta.wvp
tlsm = train.ta.stype

#%% read cases with high IWP

infile = "high_iwp_inputs_greater_120.nc"

dataset = xarray.open_dataset(infile)

tbm  = dataset.tb
miwp = dataset.iwp
mlat = dataset.lat
mlon = dataset.lon
mwvp = dataset.wvp
mt2m = dataset.t2m
mlsm = dataset.lsm

#%% extract closest database inputs 

iargs = get_args_closest_points_database(tbd[:, :].data, tbm.data)

#%%
ctb  = get_other_inputs(iargs, tbd)
cwvp = get_other_inputs(iargs, twvp)
clat = get_other_inputs(iargs, tlat)
clon = get_other_inputs(iargs, tlon)
ct2m = get_other_inputs(iargs, tt2m)
clsm = get_other_inputs(iargs, tlsm)
ciwp = get_other_inputs(iargs, tiwp)
#%%

fig, ax = plt.subplots(4, 1, figsize = [10, 30])
ax = ax.ravel()

for ix in range(4):
    ax[ix].scatter(tbm[:, ix], ctb[:, ix])
    ax[ix].set_xlabel("measurements")
    ax[ix].set_ylabel("database")
    ax[ix].set_title(freq[ix])
    

fig, ax = plt.subplots(2, 2, figsize = [10, 10])
ax = ax.ravel()

for ix, (field1, field2) in enumerate(zip([mt2m, mlat, mwvp, mlsm], 
                              [ct2m, clat, cwvp, clsm ])):   
    ax[ix].scatter(field1, field2)
    ax[ix].set_xlabel("measurements")
    ax[ix].set_ylabel("database")
    #ax[ix].set_title()

import pickle
with open("iargs.pickle", "wb") as f:
    pickle.dump(iargs, f)
    f.close()    
    






   
            
            
        
        
        
        
        


