#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:48:12 2021

@author: inderpreet
"""
import xarray
import numpy as np
from iwc2tb.GMI.write_training_data import lsm_gmi2arts

def to_dataarray_satdata(gmi):
    """
    convert GMI ARTS simulations to a dataarray

    Parameters
    ----------
    gmi : instance of GMI_Sat class

    Returns
    -------
    ta : dataarray containing ta and other paramters as attributes.

    """
    
    ta = gmi.tb.reshape(-1, 4)
    
    mask = np.isnan(ta[:, 0])
    mask = ~mask
   
    temp = np.zeros(ta[mask, :].shape)
    
    temp[:, 1] = ta[mask, 1]
    temp[:, 0] = ta[mask, 0]
    temp[:, 2] = ta[mask, 3]
    temp[:, 3] = ta[mask, 2]
    
    ta = temp.copy()
        
    
    cases = np.arange(0, ta.shape[0], 1)

    channels =   ["166.5V", "166.5H", "183+-3", "183+-7"]
    lsm = gmi.get_lsm().ravel()
    stype = lsm_gmi2arts(lsm)

    ta = xarray.DataArray(ta[mask], coords = [cases, channels], 
                          dims = ['cases', 'channels'], name = 'ta')
    ta.attrs['stype']   = stype[mask]
    ta.attrs['lon']   = gmi.lon.ravel()[mask]
    ta.attrs['lat']   = gmi.lat.ravel()[mask]

    ta.attrs['iwp']   = gmi.iwp.ravel()[mask]
    ta.attrs["rwp"]   = gmi.rwp.ravel()[mask]
    ta.attrs["t0"]    = gmi.t0.ravel()[mask]
    
    
    lst = gmi.lst.ravel()[mask]
    
    lst = [lst[i].strftime('%Y%m%d%H%M') for i in range(lst.shape[0])] 
    
    lst = np.stack(lst).T
    ta.attrs['lst']  = lst
    
    
    return ta 