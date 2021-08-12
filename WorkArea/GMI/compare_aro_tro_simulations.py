#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 07:39:46 2021

@author: inderpreet
"""


import numpy as np
import os
import xarray
import glob
import matplotlib.pyplot as plt
from iwc2tb.GMI.GMI import GMI
from matplotlib import cm
plt.rcParams.update({'font.size': 20})
#%%

def plot_tbd_pratio(tbd, bins):
    
    ilabels = np.zeros([prbins.size, bins.size])
    
    ibins   = np.digitize(tbd, bins)

    
    im1 = stype < 2
    
    for i in range(bins.size):
        
        im = ibins == i    
        im = np.logical_and(im, im1)
        ilabels[:, i] = np.bincount(ipratio[im], minlength = prbins.size)
    
    return ilabels 

#%%
def plot_tbd_surfacetype(tbd, bins, stype):
    
    hist, bin_edges = np.histogram(tbd, bins)
    
    
    ibins  = np.digitize(tbd, bins)
    
    
    
    ilabels1 = np.zeros([10, bins.size])
    
    for i in range(bins.size):
        
        im = ibins == i    
        ilabels1[:, i] = np.bincount(stype[im], minlength = 10)

    return ilabels1
#%%
def change_path_basenames(files, newpath):
    
    newfiles = []
    
    for file in files:
        basename = os.path.basename(file)
        newfile  = os.path.join(newpath, basename)
        
        newfiles.append(newfile)
        
    return newfiles  

#%%
inpath  = os.path.expanduser("~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/lpa_pr_1/")
lpapath = os.path.expanduser("~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/")


lpa_pr1_files   = glob.glob(os.path.join(inpath, "*.mat"))
lpa_files       = change_path_basenames(lpa_pr1_files, lpapath)


lpa     = GMI(lpa_files)
lpa_pr1 = GMI(lpa_pr1_files)




tb_aro = lpa.tb
tb_tro = lpa_pr1.tb


stype = lpa.stype
pr    = lpa.pratio

#%%

im1 = stype < 2
im2 = (pr >= 1.3) & (pr < 1.5 )  



im = np.logical_and(im1, im2)

#%%
tbd = tb_tro - tb_aro
#tbd = (tb_tro[:, 0] + tb_tro[:, 1])/2 - (tb_aro[:, 0] + tb_aro[:, 1])/2  

#%%

bins = np.arange(-50, 50, 1)
ilabels = np.zeros([10, bins.size, 4])

fig, ax = plt.subplots(2, 2, figsize = [12, 12])
ax = ax.ravel()

titles = ["166V GHz", "166H GHz", "183.31+-3 GHz", "183.31+-7 GHz"]


for j in range(4):
    
    ilabels[:, :, j] = plot_tbd_surfacetype(tbd[:, j], bins, stype)

    x = np.zeros(ilabels.shape[1])
    
    ilabels1 = ilabels[:, :, j]
    
    for i in range(10):
        print (i)
        x = x + ilabels1[i, :]
        if i == 0:
            ax[j].bar(bins, ilabels1[i, :]) 
        else:            
            
            ax[j].bar(bins, ilabels1[i, :], bottom = np.sum(ilabels1[:i, :], axis = 0)) 
        #ax.plot(bins, ilabels1[i, :])
        
    #ax.plot(bincenters, hist, 'o')
    #ax.plot(bins, x[:])
    ax[j].set_yscale("log")    
    
    #ax.set_yscale("log")

    ax[j].set_title(titles[j])
    
ax[0].set_ylabel("Counts")
ax[2].set_xlabel("tb_tro - tb_aro")    
ax[2].set_ylabel("Counts")
ax[3].set_xlabel("tb_tro - tb_aro") 
ax[3].legend(["water","land", "snow", "sea-ice", "coast"],bbox_to_anchor=(0.8, 1))
fig.savefig("hist_surfacetype_tro_aro.png", bbox_inches = "tight")
#-------------------------------------------------
#%%
prbins  = np.arange(1.0, 1.41, 0.05)
prbins  = np.round(prbins, decimals = 2)
ipratio = np.digitize(pr, prbins)
ilabels2 = np.zeros([prbins.size, bins.size, 4])


for j in range(4):
    ilabels2[:, :, j] = plot_tbd_pratio(tbd[:, j], bins)   
 
bincenters = 0.5*(bins[1:]+bins[:-1])
cmap = cm.get_cmap('Blues')
    
fig, ax = plt.subplots(2, 2, figsize = [12, 12])
ax = ax.ravel()

for j in range(4):

    ilabels = ilabels2[:, :, j]    
    for i in range(prbins.size):
        if i == 0:
            ax[j].bar(bincenters, ilabels[i, 1:], color = cmap(0.1 * i)) 
        else:            
            ax[j].bar(bincenters, ilabels[i, 1:], color = cmap(0.1 * i),
                   bottom = np.sum(ilabels[:i, 1:], axis = 0)) 
        
        #ax.plot(bincenters, hist)
        
    #ax.set_yscale("log")    

    ax[j].set_yscale("log")
    ax[j].set_title("water + land")
    ax[j].set_title(titles[j])
ax[0].set_ylabel("Counts")
ax[2].set_xlabel("tb_tro - tb_aro [K]")    
ax[2].set_ylabel("Counts")
ax[3].set_xlabel("tb_tro - tb_aro [K]") 
ax[3].legend(prbins)  
fig.savefig("hist_pratio_aro_water_land.png", bbox_inches = "tight")
#%%

titles = ["166V", "166H", r"183.31$\pm$3", r"183.31$\pm$7"]
mean = np.zeros([4, 4])
std  = np.zeros([4, 4])

mean[0, :] = np.mean(tbd, axis = 0)
std[0, :] = np.std(tbd, axis = 0)
for i in [0, 1, 2]:
    mask = stype == i
    mean[i+1, :] = np.mean(tbd[mask, :], axis = 0)
    std[i+1, :] = np.std(tbd[mask, :], axis = 0)
    
fig, ax = plt.subplots(1, 2, figsize = [14, 6])
ax = ax.ravel()

ax[0].plot( titles, mean[0, :], 'bD', label = "All" )
ax[0].plot( titles, mean[1, :], 'bo', label = "Water")
ax[0].plot( titles, mean[2, :], 'bx', label = "Land" )
ax[0].plot( titles, mean[3, :], 'bs', label = "Snow")
ax[0].grid('on', alpha = 0.3)
ax[0].set_ylabel("Mean [K]") 

ax[1].plot( titles, std[0, :], 'bD', label = "All" )
ax[1].plot( titles, std[1, :], 'bo', label = "Water")
ax[1].plot( titles, std[2, :], 'bx', label = "Land" )
ax[1].plot( titles, std[3, :], 'bs', label = "Snow")
ax[1].legend()
ax[1].grid('on', alpha = 0.3)
ax[1].set_ylabel("Standard deviation [K]") 
fig.savefig("std_mean.png", bbox_inches = "tight")       
    
#%%







    





