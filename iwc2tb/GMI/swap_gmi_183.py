#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:53:10 2021

@author: inderpreet
"""
import numpy as np

def swap_gmi_183(ta1):
    
        temp = np.zeros(ta1.shape)
        temp[:, 1] = ta1[:, 1]
        temp[:, 0] = ta1[:, 0]
        temp[:, 2] = ta1[:, 3]
        temp[:, 3] = ta1[:, 2]
        ta1 = temp.copy()
        
        return ta1 
