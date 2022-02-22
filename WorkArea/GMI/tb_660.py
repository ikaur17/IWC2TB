#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:55:26 2021

@author: inderpreet
"""


import numpy as np
import matplotlib.pyplot as plt
from iwc2tb.GMI.GMI import GMI
import os
from iwc2tb.common.add_gaussian_noise import add_gaussian_noise
import glob
import pandas as pd
from typhon.physics import em

plt.rcParams.update({'font.size': 22})

#%%

filepath = "/home/inderpreet/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/ici/"
files = glob.glob(os.path.join(filepath, '*.mat'))
#file2 = "/home/inderpreet/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/ici/icecube_lines/2009_001_02_A.mat"


ici = GMI(files)


tb = ici.ta

tb_noise = add_gaussian_noise(tb, [0.70, 0.65])
        

pr = ici.pratio

lat = ici.lat
lon = ici.lon

lon = lon%360

im = (lat > 40) & (lat < 65)
im1 = ((lon > 330) & (lon < 360)) | ((lon > 0) & (lon < 10)) 


im = np.logical_and(im, im1)

#%% ISMAR data

data= pd.read_csv("all_flight_664_polarisation.csv")

tbv  = data["664-V"].values
iv  = em.rayleighjeans(664.0e9, tbv)
tbv  = em.radiance2planckTb(664e9, iv)

tbh  = data["664-H"].values
ih   = em.rayleighjeans(664.0e9, tbh)
tbh  = em.radiance2planckTb(664e9, ih)

data["664-V Planck"] = tbv
data["664-H Planck"] = tbh

cols = ["#999999", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
#%%
fig, ax = plt.subplots(1, 1, figsize = [8, 8])

ax.scatter(tb_noise[:, 0], tb_noise[:, 0] - tb_noise[:, 1], s = 5, alpha = 0.8,
           label = "Global simulations", c = "lightsteelblue")#c = pr, vmin = 1, vmax = 1.4 )
ax.set_xlabel("TB 660V GHz [K]" )
ax.set_ylabel("Polarisation difference [K]")

ax.scatter(tb_noise[im, 0], tb_noise[im, 0] - tb_noise[im, 1], c = cols[0], 
           s = 5, label = "40-60N, 30W-5E ")#c = pr, vmin = 1, vmax = 1.4 )


#fig.savefig("PD_660.png", bbox_inches = "tight")






data_pd = tbv - tbh

for i, ix in enumerate(["b949", "b984", "c156", "c159", "c161"]):
    
    ima = data["flight"] == ix
    pdiff = data["664-V Planck"][ima] - data["664-H Planck"][ima] 
    ax.scatter(data["664-V Planck"][ima], pdiff,
               s = 9, c = cols[i+3], label = ix)

ax.grid("on", alpha = 0.3)    
ax.legend(markerscale = 4, framealpha = 0.2)
ax.set_ylim([-5, 20])
ax.set_xlim([100, 280])
fig.savefig("PD_660.png", bbox_inches = "tight")

#%%

bins = np.arange(80, 300, 1)

fig, ax = plt.subplots(1, 1, figsize = [8,8])

ax.hist(tb_noise[:, 0], bins, density = True, histtype= "step", label = "V")

ax.hist(tb_noise[:, 1], bins, density = True, histtype= "step", label = "H")


ax.set_yscale("log")
ax.set_xlabel("TB [K]")
ax.set_ylabel(r"PDF [K$^{-1}$]")

ax.legend()

fig.savefig("TB_660.png", bbox_inches = "tight")

#%%
mask = pr == 1
fig, ax = plt.subplots(1, 1, figsize = [8, 8])

ax.scatter(tb[:, 0], tb[:, 0] - tb[:, 1], s = 5, alpha = 0.8,
           label = "ARO", c = "lightsteelblue")#c = pr, vmin = 1, vmax = 1.4 )
ax.scatter(tb[mask, 0], tb[mask, 0] - tb[mask, 1], s = 5, alpha = 0.8,
           label = "TRO", c = "tab:red")#c = pr, vmin = 1, vmax = 1.4 )
ax.set_xlabel("TB 660V GHz [K]" )
ax.set_ylabel("Polarisation difference [K]")
fig.savefig("aro_tro_660.png")
