#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:40:50 2021

python version of atmlab function *fwhm2si* 

@author: inderpreet
"""
import numpy as np

def fwhm2si(w):
    """
    Converts FWHM to standard deviation
    The standard deviation of a normal distribution is calculated,
    calculated, based on its full width at half maximum (FWHM) .
    
    Parameters
    ----------
    w : full width at half maximum

    Returns
    -------
    si : standard deviation

    """

    si = w / (2*np.sqrt (2*np.log(2)))
    return si