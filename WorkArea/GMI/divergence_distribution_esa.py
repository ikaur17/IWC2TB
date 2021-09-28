#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:15:34 2021

@author: inderpreet
"""

import numpy as np
import os
import glob
from iwc2tb.GMI.GMI import GMI
from iwc2tb.GMI.GMI_SatData import GMI_Sat
import random
import scipy
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from scipy.ndimage.filters import gaussian_filter
plt.rcParams.update({'font.size': 20})

#%%
def get_hist2d(ta, tb_gmi):

    # ybins1  = np.arange(-5, 5, 0.5)
    # ybins2 = np.arange(5, 20, 0.75)
    # ybins3 = np.arange(20, 30, 1.25)
    # ybins4 = np.arange(30, 60, 2)
    # ybins = np.concatenate([ybins1, ybins2, ybins3, ybins4])
    counts, _, _  = np.histogram2d(ta[:, 0], ta[:, 0] - ta[:, 1],  
                                        bins=(xbins, ybins))
    
    counts1, _, _ = np.histogram2d(tb_gmi[:, 0].ravel(), 
                                         tb_gmi[:, 0].ravel() - tb_gmi[:,  1].ravel(),  
                                        bins=(xbins, ybins))

    return counts, counts1



#%%
    
inpath_gmi   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2020/01/')
# GMI simulations    
gmifiles = glob.glob(os.path.join(inpath_gmi, "*/*.HDF5"))

random.shuffle(gmifiles)
gmi_sat = GMI_Sat(gmifiles[:20])


key = "lpa"
# GMI frquencies
freq     = ["166.5V", "166.5H",  "183+-7", "183+-3",]   

cols = ["#999999", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

tb_gmi  = gmi_sat.tb

#%%
inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/esa') 
matfiles1 = glob.glob(os.path.join(inpath_mat, "2009_00*.mat"))
matfiles2 = glob.glob(os.path.join(inpath_mat, "2009_01*.mat"))
matfiles3 = glob.glob(os.path.join(inpath_mat, "2009_02*.mat"))
matfiles  = matfiles1 + matfiles2 + matfiles3

gmi                     = GMI(matfiles[:])
ta_aro                  = gmi.ta_noise

#%%

inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/esa_pr_1') 
matfiles1 = glob.glob(os.path.join(inpath_mat, "2009_00*.mat"))
matfiles2 = glob.glob(os.path.join(inpath_mat, "2009_01*.mat"))
matfiles3 = glob.glob(os.path.join(inpath_mat, "2009_02*.mat"))
matfiles  = matfiles1 + matfiles2 + matfiles3
gmi_tro                 = GMI(matfiles[:])
ta_tro                  = gmi_tro.ta_noise


#%%

mask = (gmi.pratio > 1.25) & (gmi.pratio < 1.35) 
ta_13 = ta_aro[mask]


mask = (gmi.pratio > 1.15) & (gmi.pratio < 1.25) 
ta_12 = ta_aro[mask]

#%%
from scipy.stats import truncnorm
prbins = np.arange(1, 1.45, 0.05)
pr     = gmi.pratio

ndist    = np.random.normal(1.2, 0.24, 100000)

lower, upper = 1, 1.4
mu, sigma = 1.2, 0.24
X = truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

ndist = X.rvs(100000)

nhist, _ = np.histogram(ndist, prbins)

nratio   = nhist/nhist.max()

ipr     = np.digitize(pr, prbins)
icounts = np.bincount(ipr)

ncounts = (icounts[1:] * nratio).astype(int)

ta_normal = []


for i in range(8):

         print (i)    

         pr_subset = ta_aro[ipr == i+1, :] 
        
         args = np.arange(0, ncounts[i], 1)
    
         nargs = np.random.choice(args, size = ncounts[i])
         
         print (nargs.size, ncounts[i], pr_subset.shape)

         ta_normal.append(pr_subset[nargs, :])

        
ta_normal = np.concatenate(ta_normal, axis = 0)  

#%%
xbins = np.arange(100, 240, 2)
ybins = np.arange(-5, 25, 1.5)

xcenter = (xbins[1:] + xbins[:-1])/2
ycenter = (ybins[1:] + ybins[:-1])/2

colors = iter([plt.cm.Dark2(i) for i in range(6)])



fig, ax = plt.subplots(1, 2, figsize = [16, 8])
ax.ravel()

sigma = 1.0

# TRO
im  = tb_gmi[:, :, 0] < 240
im1 = ta_tro[:, 0] < 240
counts_tro, counts_gmi = get_hist2d(ta_tro[im1, :], tb_gmi[im, :])
a11 = scipy.spatial.distance.jensenshannon( counts_tro, counts_gmi)
a12 = scipy.spatial.distance.jensenshannon( counts_tro.T, counts_gmi.T)

cc = cols[0]
a12 = gaussian_filter(a12, sigma)
a11 = gaussian_filter(a11, sigma)
ax[0].plot(xcenter, a12**2, label = r"$\rho$ = 1", c= cc, linewidth = 2.5)
ax[1].plot(ycenter, a11**2, label =r"$\rho$ = 1", c= cc, linewidth = 2.5)


# # 1.2
im  = tb_gmi[:, :, 0] < 240
im1 = ta_12[:, 0] < 240
counts_tro, counts_gmi = get_hist2d(ta_12[im1, :], tb_gmi[im, :])
a11 = scipy.spatial.distance.jensenshannon( counts_tro, counts_gmi)
a12 = scipy.spatial.distance.jensenshannon( counts_tro.T, counts_gmi.T)

cc = cols[1]
a12 = gaussian_filter(a12, sigma)
a11 = gaussian_filter(a11, sigma)
ax[0].plot(xcenter, a12**2, label = r"$\rho$ = 1.2", c= cc, linewidth = 2.5)
ax[1].plot(ycenter, a11**2, label =r"$\rho$ = 1.2", c= cc, linewidth = 2.5)


# # 1.3
im  = tb_gmi[:, :, 0] < 240
im1 = ta_13[:, 0] < 240
counts_tro, counts_gmi = get_hist2d(ta_13[im1, :], tb_gmi[im, :])
a11 = scipy.spatial.distance.jensenshannon( counts_tro, counts_gmi)
a12 = scipy.spatial.distance.jensenshannon( counts_tro.T, counts_gmi.T)

cc = cols[2]
a12 = gaussian_filter(a12, sigma)
a11 = gaussian_filter(a11, sigma)
ax[0].plot(xcenter, a12**2, label = r"$\rho$ = 1.3", c= cc, linewidth = 2.5)
ax[1].plot(ycenter, a11**2, label =r"$\rho$ = 1.3", c= cc, linewidth = 2.5)




# ARO
im  = tb_gmi[:, :,  0] < 240
im1 = ta_aro[:, 0] < 240
counts_aro, counts_gmi = get_hist2d(ta_aro[im1, :], tb_gmi[im, :])
a1 = scipy.spatial.distance.jensenshannon(counts_aro, counts_gmi)
a2 = scipy.spatial.distance.jensenshannon(counts_aro.T, counts_gmi.T)

cc = cols[3]
a2 = gaussian_filter(a2, sigma)
a1 = gaussian_filter(a1, sigma)
ax[0].plot(xcenter, a2**2, label = r"$\rho\in$ U(1, 1.4)",  color =  cc, linewidth = 2.5)
ax[1].plot(ycenter, a1**2, label = r"$\rho\in$ U(1, 1.4)",  color = cc, linewidth = 2.5)


# ARO
im  = tb_gmi[:, :,  0] < 240
im1 = ta_normal[:, 0] < 240
counts_aro, counts_gmi = get_hist2d(ta_normal[im1], tb_gmi[im])
#a1 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 0)
#a2 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 1)

a1 = scipy.spatial.distance.jensenshannon(counts_aro, counts_gmi)
a2 = scipy.spatial.distance.jensenshannon(counts_aro.T, counts_gmi.T)

cc = cols[4]

a2 = gaussian_filter(a2, sigma)
a1 = gaussian_filter(a1, sigma)
ax[0].plot(xcenter, a2**2, label = r"$\rho\in$ N(1.2, 0.12)",  color =  cc, linewidth = 2.5)
ax[1].plot(ycenter, a1**2, label =  r"$\rho\in$ N(1.2, 0.12)",  color = cc, linewidth = 2.5)



ax[0].set_xlabel("TB 166V GHz [K]")
ax[1].set_xlabel("Polarisation difference [K]")

ax[0].set_ylabel("Divergence")
ax[1].set_ylabel("Divergence")

for i in range(2):
    ax[i].grid("on", alpha = 0.3)

ax[1].legend()
    
fig.savefig("divergence_esa.pdf", bbox_inches = "tight")  


            
         
         

  


#%%

xbins = np.arange(100, 300, 5)
ybins = np.arange(-5, 60, 2.5)

fig, ax = plt.subplots(1, 3, figsize = [24, 8])
ax = ax.ravel()

colors = iter([plt.cm.tab20(i) for i in range(6)])

c1 = [next(colors)]
c2 = [next(colors)]


im  = tb_gmi[:, :, 0] < 240


ax[0].scatter(tb_gmi[im, 0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im,  1].ravel()[::10], 
           label = 'Observation', c = c2, s = 2)




counts, xbins, ybins=np.histogram2d(tb_gmi[im, 0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im, 1].ravel()[::10],  
                                        bins=(xbins, ybins))



sigma = 0.05 # this depends on how noisy your data is, play with it!

ax[0].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'black', locator=ticker.LogLocator(), alpha = 0.5)


im1 = ta_tro[:, 0] < 240
counts, xbins, ybins=np.histogram2d(ta_tro[im1, 0], ta_tro[im1, 0] - ta_tro[im1, 1],  
                                        bins=(xbins, ybins))

counts = gaussian_filter(counts, sigma)
ax[0].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'red', locator=ticker.LogLocator(), alpha = 0.5)


#ax[0].scatter(ta_tro[:, 0][::10], ta_tro[:, 0][::10] - ta_tro[:, 1][::10], 
#           label = r"Simulation $\rho$ = 1" , c = c2, s = 2)

cs_sc = ax[1].scatter(tb_gmi[im, 0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im, 1].ravel()[::10], 
           label = 'Observation', c = c2, s = 2)





counts, xbins, ybins=np.histogram2d(tb_gmi[im, 0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im, 1].ravel()[::10],  
                                        bins=(xbins, ybins))

counts = gaussian_filter(counts, sigma)
ax[1].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'black', locator=ticker.LogLocator(), 
                alpha = 0.5, label = "Observation")

counts, xbins, ybins=np.histogram2d(ta_tro_12[:, 0], ta_tro_12[:, 0] - ta_tro_12[:, 1],  
                                        bins=(xbins, ybins))
counts = gaussian_filter(counts, sigma)
ax[1] .contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'red', locator=ticker.LogLocator(), 
                alpha = 0.5, label =   r"Simulation [\rho = 1]")



ax[2].scatter(tb_gmi[:,:, 0].ravel()[::10], 
           tb_gmi[:, :,0].ravel()[::10] - tb_gmi[:, :,  1].ravel()[::10], 
           label = 'Observation', c = c2, s = 2)


counts, xbins, ybins=np.histogram2d(tb_gmi[:,:, 0].ravel()[::10], 
           tb_gmi[:, :,0].ravel()[::10] - tb_gmi[:, :,  1].ravel()[::10],  
                                        bins=(xbins, ybins))

counts = gaussian_filter(counts, sigma)
cs_gmi = ax[2].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'black', locator=ticker.LogLocator(),
                alpha = 0.5, label = r"Simulation [\rho = 1.2]")

counts, xbins, ybins=np.histogram2d(ta_aro[:, 0], ta_aro[:, 0] - ta_aro[:, 1],  
                                        bins=(xbins, ybins))

counts = gaussian_filter(counts, sigma)
cs = ax[2].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'red', locator=ticker.LogLocator(),
                alpha = 0.5, label = r"Simulation [\rho \in U[1, 1.4]]")


lines = [ cs.collections[0], cs_gmi.collections[0]]


#ax[1].scatter(ta_tro_13[:, 0][::5], ta_tro_13[:, 0][::5] - ta_tro_13[:, 1][::5], 
#           label = r"Simulation $\rho$ = 1.3" , c = c2, s = 2)

for i in range(3):
    ax[i].grid("on", alpha = 0.3)
    ax[i].set_xlabel("TB 166V GHz [K]")
    ax[0].set_ylabel("Polarisation difference [K]")
    lgnd = ax[i].legend()
    ax[i].legend(lines, [ "Simulation", "Observation"], loc = 'upper left')
#change the marker size manually for both lines
    lgnd.legendHandles[0]._sizes = [30]
#    lgnd.legendHandles[1]._sizes = [30]

ax[0].set_title(r"$\rho$ = 1")
ax[1].set_title(r"$\rho$ = 1.2")
ax[2].set_title(r"$\rho$ $\in$ U(1, 1.4)")
fig.savefig("PD_166.png", bbox_inches = "tight")
#cc = next(colors)
#ax.scatter(ta_tro_12[:, 0], ta_tro_12[:, 0] - ta_tro_12[:, 1], label = r"$\rho$ = 1.2", c = cc )
    
    
    


