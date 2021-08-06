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


def change_path_basenames(files, newpath):
    
    newfiles = []
    
    for file in files:
        basename = os.path.basename(file)
        newfile  = os.path.join(newpath, basename)
        
        newfiles.append(newfile)
        
    return newfiles  


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



im1 = stype < 2
im2 = (pr >= 1.3) & (pr < 1.5 )






    



im = np.logical_and(im1, im2)




tbd = (tb_tro[:, 0] - tb_tro[:, 1]) - (tb_aro[:, 0] - tb_aro[:, 1])  


#%%
tbd = tb_tro[:, 0] - tb_aro[:, 0]

bins = np.arange(-80, 80, 1)

bincenters = 0.5*(bins[1:]+bins[:-1])

cmap = cm.get_cmap('Blues')
hist, bin_edges = np.histogram(tbd, bins)


ibins  = np.digitize(tbd, bins)



ilabels1 = np.zeros([10, bins.size])

for i in range(bins.size):
    
    im = ibins == i    
    ilabels1[:, i] = np.bincount(stype[im], minlength = 10)
    



fig, ax = plt.subplots(1, 1, figsize = [12, 6])

x = np.zeros(ilabels1.shape[1])
for i in range(10):
    print (i)
    x = x + ilabels1[i, :]
    if i == 0:
        ax.bar(bins, ilabels1[i, :]) 
    else:            
        
        ax.bar(bins, ilabels1[i, :], bottom = np.sum(ilabels1[:i, :], axis = 0)) 
    #ax.plot(bins, ilabels1[i, :])
    
#ax.plot(bincenters, hist, 'o')
#ax.plot(bins, x[:])
ax.set_yscale("log")    
ax.legend(["water","land", "snow", "sea-ice", "coast"])
#ax.set_yscale("log")
ax.set_ylabel("Counts")
ax.set_xlabel("tb_tro - tb_aro 166V GHz [K]")
ax.set_title("all surface types")
fig.savefig("hist_surfacetype_tro_aro.png", bbox_inches = "tight")
#-------------------------------------------------
#%%
prbins = np.arange(1.0, 1.41, 0.05)
prbins = np.round(prbins, decimals = 2)
ipratio = np.digitize(pr, prbins)
ilabels = np.zeros([prbins.size, bins.size])



im1 = stype < 2

for i in range(bins.size):
    
    im = ibins == i    
    im = np.logical_and(im, im1)
    ilabels[:, i] = np.bincount(ipratio[im], minlength = prbins.size)
 
fig, ax = plt.subplots(1, 1, figsize = [12, 6])

for i in range(prbins.size):
    if i == 0:
        ax.bar(bincenters, ilabels[i, 1:], color = cmap(0.1 * i)) 
    else:            
        ax.bar(bincenters, ilabels[i, 1:], color = cmap(0.1 * i),
               bottom = np.sum(ilabels[:i, 1:], axis = 0)) 
    
    #ax.plot(bincenters, hist)
    
#ax.set_yscale("log")    
ax.legend(prbins)
ax.set_yscale("log")
ax.set_ylabel("Counts")
ax.set_xlabel("tb_tro - tb_aro 166V GHz [K]")
ax.set_title("water + land")
fig.savefig("hist_pratio_aro_water_land.png", bbox_inches = "tight")
#%%







    





