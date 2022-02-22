#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 21:46:42 2021

@author: inderpreet
"""


import xarray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

a = xarray.open_dataset("jan2020_IWP_he_loglinear.nc")
b = xarray.open_dataset("jan2020_IWP_he_log.nc")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = [6, 18])

mask = b.iwp_mean.data > 15

cs = ax1.scatter(a.iwp_mean.data[mask], b.iwp_mean.data[mask],
                c = a.lsm.data[mask], cmap = cm.tab20)

ax1.set_xlabel("IWP (Loglinear)")
ax1.set_ylabel("IWP (Log)")

mask = np.logical_and(b.iwp_mean.data > 5 , b.iwp_mean.data < 15)

cs = ax2.scatter(a.iwp_mean.data[mask], b.iwp_mean.data[mask],
                c = a.lsm.data[mask], cmap = cm.tab20)

ax2.set_xlabel("IWP (Loglinear)")
ax2.set_ylabel("IWP (Log)")


mask = np.logical_and(b.iwp_mean.data > 1 , b.iwp_mean.data < 5)
cs = ax3.scatter(a.iwp_mean.data[mask], b.iwp_mean.data[mask],
                c = a.lsm.data[mask], cmap = cm.tab20)

fig.colorbar(cs, ax = (ax1, ax2, ax3))
ax3.set_xlabel("IWP (Loglinear)")
ax3.set_ylabel("IWP (Log)")

mask = b.iwp_mean.data < 1

bins = np.array([0.0001, 0.00025, 0.00050, 0.00075, 0.001, 0.0025, 
                 0.0050, 0.0075, 0.01, 0.025, 0.050, 0.075, 
                 0.1, 0.25, 0.50, 0.75, 1, 5, 10]) 

ax4.hist(a.iwp_mean.data[mask], bins, density = True, label = "LogLinear", histtype = "step")
ax4.hist(b.iwp_mean.data[mask], bins, density = True, label = "Log", histtype = "step") 
ax4.set_yscale("log")
ax4.set_xscale("log")
ax4.set_xlabel("IWP")
ax4.set_ylabel("PDF")
ax4.legend()

fig.savefig("log_loglinear_stype.png", bbox_inches = "tight")
