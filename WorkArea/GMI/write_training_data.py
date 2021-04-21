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
    ta.attrs["t2m"]   = gmi.t2m
    ta.attrs["skt"]   = gmi.skt
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
    ta.attrs["wvp"]   = gmi.wvp.ravel()[mask]
    
    
    lst = gmi.lst.ravel()[mask]
    
    lst = [lst[i].strftime('%Y%m%d%H%M') for i in range(lst.shape[0])] 
    
    lst = np.stack(lst).T
    ta.attrs['lst']  = lst
    
    
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
    
    inpath    =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_testsimulations/test_f07')  
    inpath1   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_testsimulations/test_si') 
    inpath    =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65/')
 
     
    matfiles  = glob.glob(os.path.join(inpath, "2010_*.mat"))
    #matfiles1 = glob.glob(os.path.join(inpath1, "2010_*.mat"))
    
    #matfiles += matfiles1
    
    gmi       = GMI(matfiles)
#%%    
    ta        = to_dataarray(gmi)
    cases     = ta.cases.values 
    
    randomList = random.sample(range(0, len(cases)), len(cases))
    lim       = int(len(cases) * 0.05)
    
#%%    
    ta_test = divide_test_train(ta, randomList[:lim])
    ta_test.to_netcdf('TB_GMI_test_multiple_t.nc', 'w')
    
    ta_train = divide_test_train(ta, randomList[lim:])
    ta_train.to_netcdf('TB_GMI_train_multiple_t.nc', 'w')
    
#%%
    inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2021/19')
    inpath   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2019/01/01')
    gmifiles = glob.glob(os.path.join(inpath, "*.HDF5"))

    gmi_s    = GMI_Sat(gmifiles[2:3])
#%%    
    ta_s      = to_dataarray_satdata(gmi_s)

    
    ta_s.to_netcdf('TB_GMI_test_satdata.nc', 'w')
#%%    
    
    
