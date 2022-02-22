#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 09:54:43 2022

@author: inderpreet
"""

import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator


def delanuay_interp(
        xp: np.ndarray,
        fp: np.ndarray,
        x: np.ndarray
        ):
    """
    

    Parameters
    ----------
    xp : array_like, The coordinates of the data points.
    fp : array_like, The function evaluated at points xp, same length as xp.
    x  : array_like, The coordinates at which to evaluate the interpolated values.

    Returns
    -------
    yinterp : array_like, the interpolated values at coordinates given by x

    """

    #
    tri = Delaunay(xp)
    
    interp = LinearNDInterpolator(tri, fp)
    #
    yinterp = interp(x)
    return yinterp
    
    
        