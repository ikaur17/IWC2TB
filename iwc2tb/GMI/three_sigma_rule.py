#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:04:18 2021

@author: inderpreet
"""
import numpy as np

def three_sigma_rule(ta, bins = None):
    if bins == None:
        bins = np.arange(100, 300, 0.1)
    hist = np.histogram(ta, bins, density = True)
    
    ix  = np.argmax(hist[0])
    TB0 = (bins[ix] + bins[ix+1])/2
    sig = np.std(hist[0])
    
    Tx = TB0 - 3 * sig 

    return Tx    
