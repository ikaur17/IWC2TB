#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:52:34 2021

@author: inderpreet
"""

import glob
import os
import numpy as np
from era2dardar.RADARLIDAR import DARDAR
import pickle


#----------------------------------------------
outfile = "dardar_jan2009"
year = "2009"
month = "01"
#----------------------------------------------

dfiles = []
#dpath  = "/home/inderpreet/Dendrite/SatData/DARDAR/2008/12/"
#dfiles += glob.glob(os.path.join(dpath + "*/*.hdf"))  

dpath  = os.path.join("/home/inderpreet/Dendrite/SatData/DARDAR/", year, month)
dfiles += glob.glob(os.path.join(dpath + "*/*.hdf"))  

#dpath  = "/home/inderpreet/Dendrite/SatData/DARDAR/2009/02/"
#dfiles += glob.glob(os.path.join(dpath + "*/*.hdf"))  

DIWP = []
DLAT = []
DLON = []
for dfile in dfiles:
    dardar =  DARDAR(dfile)
    print (dfile)
    
    diwc    = dardar.iwc
    dlat    = dardar.latitude
    dlon    = dardar.longitude
    dh      = dardar.height

    dmask   = np.abs(dlat) <= 65.0
    diwp    = np.zeros(dlat[dmask].shape[0])    
    for i in range(diwc[dmask, :].shape[0]):
        diwp[i] = np.trapz(diwc[i, :], dh)
    

    DIWP.append(diwp)
    DLAT.append(dlat[dmask])
    DLON.append(dlon[dmask])

diwp = -1 * np.concatenate(DIWP)
dlat = np.concatenate(DLAT)
dlon = np.concatenate(DLON)



with open(outfile + ".pickle", "wb") as f:
    pickle.dump(dlat, f)
    pickle.dump(dlon, f)
    pickle.dump(diwp, f)
f.close()


#%%