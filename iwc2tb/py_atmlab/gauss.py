#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:43:12 2021

python version of atmlab *gauss* function

@author: inderpreet
"""

import numpy as np

def  gauss(x,s,m):
    """
    The Gauss function
    Calculates the scalar Gauss function:
    y = 1/(s*sqrt(2*pi)) * exp(-0.5((x-m)/(s))^2)
    where x is a vector.

#% 2006-03-02   Created by Stefan Buehler

    Parameters
    ----------
    x : input vector
    s : standard deviation
    m : mean

    Returns
    -------
    y : function values

    """

    y = 1.0 / ( s * np.sqrt(2.0 * np.pi) ) * np.exp( -0.5 * ((x-m)/s)**2 )
    return y