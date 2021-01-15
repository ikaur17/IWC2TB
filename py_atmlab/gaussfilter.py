#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:57:52 2021

python version of *gaussfilter* matlab function available as part of atmlab package
# % 2007-11-29   Created by Patrick Eriksson.

@author: inderpreet
"""

import numpy as np
from fwhm2si import fwhm2si
from gauss import gauss


def issorted(x):
    """
    checks if the values are  in sorted order

    Parameters
    ----------
    x : array to be checked

    Returns
    -------
    boolean : True for sorted, False otherwise
    """
    return (np.arange(len(x)) == np.argsort(x)).all()


def gaussfilter(x,y,xw):
    """
# %    The data are assumed to be equidistant, but gaps in the data series are
# %    allowed. Parts of the filter extending outside the data range are
# %    neglected. The response is normalised for each point and if y equals
# %    one everywhere, also yf will be one everywhere.
# %
# %
# %        
# % OUT   yf   Filtered data.
# % IN    x    Data abscissa.
# %       y    Data values. Must be given as a column vector or a matrix.
# %       xw   Width (total) of the filter. 

# % 2006-04-05   Created by Patrick Eriksson.

    Parameters
    ----------
    x : data abscissa
    y : Data values. Must be given as a column vector or a matrix.
    xw : Width (total) of the filter

    Raises
    ------
    Exception
        1. if size of y.shape[0] and x do not match   
        2. if x is not sorted

    Returns
    -------
    yf : np.array of filtered data

    """
    if np.shape(y)[0] != len(x):
       raise Exception('Size of *y* does not match size of *x*.')
 
    if ~issorted(x):
      raise Exception('The vector *x* must be sorted.')
    
    
# Set-up output variable
    yf = y.copy()
        
    
# Calculate "1 std dev"
    si = fwhm2si( xw )    
    
    for i in range(len(x)):

# Calculate distance and determine data points to consider 
        
      d       = abs( x - x[i] );
      ind     = np.where( d < si*4 )[0]
      
# Calculate weights
      w = gauss( x[ind], si, x[i] )

# Filter
      yf[i,:] = np.sum(y[ind,:] * np.tile(w, (1, np.shape(y)[1])).T) / np.sum(w)
      
    return yf
