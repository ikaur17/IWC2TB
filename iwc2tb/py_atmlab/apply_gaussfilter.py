#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:47:59 2021

@author: inderpreet
"""
import numpy as np
from iwc2tb.py_atmlab .gaussfilter import gaussfilter
import matplotlib.pyplot as plt

def apply_gaussfilter(x, y, w):   
    
    x2, i2, i1 = np.unique(x, return_index=True, return_inverse=True)
    
    y = y.reshape(-1, 1)
    yf = gaussfilter(x2, y[i2], w)
    
    yfilter = yf[i1]
    fig, ax = plt.subplots(1, 1, figsize = [8,8])
    ax.plot(x, y, label = "tb")
    ax.plot(x, yfilter, label = "ta")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()   
    plt.show()     
    return yfilter