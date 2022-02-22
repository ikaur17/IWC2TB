#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:06:29 2021

@author: inderpreet
"""
import numpy as np
import matplotlib.pyplot as plt



def plot_pdf_gmi(Ta, Tb, bins= None, figname = "distribution_gmi.pdf"):
    """
    

    Parameters
    ----------
    Ta : simulated GMI antenna TB
    Tb : GMI TB
    bins : TYPE, optional
        DESCRIPTION. The default is None.
    figname : TYPE, optional
        DESCRIPTION. The default is "distribution_gmi.pdf".

    Returns
    -------
    None.

    """
           
    # GMI frequencies
    freq     = ["166.5V", "166.5H", "183+-3", "183+-7"]
    
    if bins is None:
        bins = np.arange(100, 310, 2)
        
    fig, axs = plt.subplots(1,2, figsize = [16, 8])
    fig.tight_layout(pad=3.0)
    
    for i, ax in enumerate(fig.axes):
        
        if i == 1:
            i = 2
    
        hist_a = np.histogram(Ta[:, i],  bins, density = True)      
        hist_b = np.histogram(Tb[:, i],  bins, density = True)  
        
        ax.plot(bins[:-1], hist_b[0],'b', label =  freq[i]+ ' obs', linewidth = 2, alpha = 0.5)       
        ax.plot(bins[:-1], hist_a[0],'b--', label =  freq[i] + ' sim', linewidth = 2, alpha = 0.5)
        
        hist_a = np.histogram(Ta[:, i+1],  bins, density = True)      
        hist_b = np.histogram(Tb[:, i+1],  bins, density = True)  
        
        ax.plot(bins[:-1], hist_b[0], 'r', label =  freq[i+1] + ' obs', linewidth = 2, alpha = 0.5)       
        ax.plot(bins[:-1], hist_a[0], 'r--', label =  freq[i+1] +  ' sim', linewidth = 2, alpha = 0.5)

#        ax.set_title(freq[i] + " GHz")
    
        ax.set_yscale('log')
        ax.set_ylabel('PDF [#/K]')
        ax.set_xlabel('Ta [K]')
        ax.legend()
        ax.grid("on", alpha = 0.4)
    fig.savefig("Figures/" + figname, bbox_inches = "tight")
