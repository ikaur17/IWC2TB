#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 21:26:04 2021

@author: inderpreet
"""
import xarray
import numpy as np
import os
import glob
from iwc2tb.GMI.GMI import GMI
import random
from iwc2tb.GMI.GMI_SatData import GMI_Sat
#%%
def to_dataarray(gmi):
    """
    convert GMI ARTS simulations to a dataarray

    Parameters
    ----------
    gmi : instance of GMI class

    Returns
    -------
    ta : dataarray containing ta and other paramters as attributes.

    """
    
    ta = gmi.ta
    cases = np.arange(0, ta.shape[0], 1)

    channels =   ["166.5V", "166.5H", "183+-3", "183+-7"]
    
    ta = xarray.DataArray(ta, coords = [cases, channels], dims = ['cases', 'channels'], name = 'ta')
    ta.attrs['stype'] = gmi.stype
    ta.attrs['lon']   = gmi.lon
    ta.attrs['lat']   = gmi.lat
    ta.attrs['iwp']   = gmi.iwp
    ta.attrs['wvp']   = gmi.wvp
    ta.attrs["rwp"]   = gmi.rwp
    ta.attrs["t0"]    = gmi.t0
    ta.attrs["p0"]    = gmi.p0
    ta.attrs["z0"]    = gmi.z0   
    
    return ta

#%%
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
    cases = np.arange(0, ta.shape[0], 1)

    channels =   ["166.5V", "166.5H", "183+-3", "183+-7"]
    lsm = gmi.get_lsm().ravel()
    stype = lsm_gmi2arts(lsm)

    ta = xarray.DataArray(ta, coords = [cases, channels], dims = ['cases', 'channels'], name = 'ta')
    ta.attrs['stype']   = stype
    ta.attrs['lon']   = gmi.lon.ravel()
    ta.attrs['lat']   = gmi.lat.ravel()
    ta.attrs['iwp']   = gmi.get_gprofdata("iceWaterPath").ravel()
    #ta.attrs['wvp']  = gmi.get_gprofdata("")
    ta.attrs["rwp"]   = gmi.get_gprofdata("rainWaterPath").ravel()
    ta.attrs["t0"]    = gmi.get_gprofdata("temp2mIndex").ravel()
    #ta.attrs["p0"]   = gmi.p0
    #ta.attrs["z0"]   = gmi.z0   
    
    return ta 

#%%
def lsm_gmi2arts(lsm):
        
    stype = lsm.copy()
    stype[lsm == 1] = 0
    stype[lsm == 2] = 3
    stype[lsm == 3] = 1
    stype[lsm == 4] = 1
    stype[lsm == 5] = 1
    stype[lsm == 6] = 1
    stype[lsm == 7] = 1
    stype[lsm == 8] = 2
    stype[lsm == 9] = 2
    stype[lsm == 10] = 2
    stype[lsm == 11] = 2
    stype[lsm == 12] = 0
    stype[lsm == 13] = 4
    stype[lsm == 14] = 6
    return stype        
        
#%%
def divide_test_train(TB, randomList):
    """
    Divide the dataarray to  a smaller dataarray according to coordinates 
    given in randomList

    Parameters
    ----------
    TB : dataarray
    randomList : list of coordinates

    Returns
    -------
    TB_test : the subsampled dataarray according to randomList

    """
    
    TB_test = TB[randomList, :]
    
    for attr in TB.attrs:
        print (attr)
        TB_test.attrs[attr] = TB.attrs[attr][randomList]        
        
    return TB_test    

#%%
if __name__ == "__main__":
    
    inpath    =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test_f07')  
    inpath1   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test1.2') 
 
     
    matfiles  = glob.glob(os.path.join(inpath, "2010_*.mat"))
    matfiles1 = glob.glob(os.path.join(inpath1, "2010_*.mat"))
    
    matfiles += matfiles1
    
    gmi       = GMI(matfiles)
#%%    
    ta        = to_dataarray(gmi)
    cases     = ta.cases.values 
    
    randomList = random.sample(range(0, len(cases)), len(cases))
    lim       = int(len(cases) * 0.32)
    
#%%    
    ta_test = divide_test_train(ta, randomList[:lim])
    ta_test.to_netcdf('TB_GMI_test.nc', 'w')
    
    ta_train = divide_test_train(ta, randomList[lim:])
    ta_train.to_netcdf('TB_GMI_train.nc', 'w')
    
#%%
    inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2019/01/')
    #inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2019/06/')
    gmifiles = glob.glob(os.path.join(inpath, "*/*20190124-S125318*.HDF5"))

    gmi_s    = GMI_Sat(gmifiles)
#%%    
    ta_s        = to_dataarray_satdata(gmi_s)
    cases     = ta.cases.values 

    ta_s.to_netcdf('TB_GMI_test_satdata.nc', 'w')
