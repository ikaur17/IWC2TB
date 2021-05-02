#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:11:01 2021

@author: inderpreet
"""
import matplotlib.pyplot as plt

def plot_scatter(Ta, Tb_gmi,  freq, figname = "scatter_gmi.pdf"):
    fig, axs = plt.subplots(2, 3, figsize = [30, 20])
    fig.tight_layout(pad=3.0)
    for ix, i in enumerate(range(0, 3)):

        for jx, j in enumerate(range(i+1, 4)):

            if ix == 2:
                jx = 2
                ix = 1
   
#            axs[ix, jx].scatter(Ta[:, i], Ta[:, j], label = 'simulated', alpha = 0.1)
            axs[ix, jx].scatter(Tb_gmi[:, i], Tb_gmi[:, j], label = 'observed', alpha = 0.3) 
            axs[ix, jx].scatter(Ta[:, i], Ta[:, j], label = 'simulated', alpha = 0.3)

            axs[ix, jx].set_xlabel(freq[i] + " [GHz]")
            axs[ix, jx].set_ylabel(freq[j] + " [GHz]")
            axs[ix, jx].legend()
    
    fig.savefig("Figures/" + figname, bbox_inches = "tight" )      