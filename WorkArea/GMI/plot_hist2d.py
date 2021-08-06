#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:12:30 2021

@author: inderpreet
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

def plot_hist2d(ta, tb0, figname = "hist2d.png"):
    
    pd = ta[:, 0] - ta[:, 1]
    
    pd_sampled = pd[::3]
    ta_sampled = ta[::3, 0]
    
    pd_sampled_gmi = (tb0[:, 0] - tb0[:, 1])
    ta_sampled_gmi = tb0[:, 0]
    
    fig, ax = plt.subplots(1, 1, figsize = [15, 15])
    xbins = np.arange(100, 310, 2)
    ybins = np.arange(-5, 60, 1)
    # ybins1  = np.arange(-5, 5, 0.5)
    # ybins2 = np.arange(5, 20, 0.75)
    # ybins3 = np.arange(20, 30, 1.25)
    # ybins4 = np.arange(30, 60, 2)
    # ybins = np.concatenate([ybins1, ybins2, ybins3, ybins4])
    counts, xbins, ybins=np.histogram2d(ta_sampled, pd_sampled,  
                                        bins=(xbins, ybins))
# make the contour plot

    cs = (ax.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'blue', locator=ticker.LogLocator(), alpha = 0.5))




    
        
    counts, xbins, ybins=np.histogram2d(ta_sampled_gmi, pd_sampled_gmi,
                                        bins=(xbins, ybins))
# make the contour plot

    cs_gmi  = (ax.contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'red',locator=ticker.LogLocator(),  alpha = 0.5))
    
    lines = [ cs.collections[0], cs_gmi.collections[0]]
#    labels = ['CS1_neg','CS1_pos','CS2_neg','CS2_pos']
    plt.legend(lines, ["simulated", "observed"], loc = 'upper left')
    ax.set_xlabel(" Brightness temperature 166 V [K] ")
    ax.set_ylabel("Polarisation difference [V-H] [K]")

    # fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    # ax.scatter(ta_sampled_gmi, pd_sampled_gmi, label = "observed", alpha = 0.3)
    # ax.scatter( ta_sampled, pd_sampled, label = "simulated", alpha = 0.3)

    # ax.set_xlabel(" Brightness temperature 166 V [K] ")
    # ax.set_ylabel("Polarisation difference [V-H] [K]")
    # ax.legend()

    fig.savefig("Figures/" + figname, bbox_inches = "tight")