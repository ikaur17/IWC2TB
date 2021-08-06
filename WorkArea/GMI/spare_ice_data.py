#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:49:49 2021

@author: inderpreet
"""


from iwc2tb.GMI.spareice import spareICE
import glob
import os
import pickle
import numpy as np

#-----------------------------------------
outfile = "spareice_jul2009"
year = "2009"
month = "06"

#-----------------------------------------

sfiles = []
spath = os.path.join("/home/inderpreet/Dendrite/SatData/SpareICE/avhrr_345_mhs_345_angles_tsurfcfsr_all_global_h/", 
                     year , month)
sfiles += glob.glob(os.path.join(spath , "*/*.gz"))

# spath = "/home/inderpreet/Dendrite/SatData/SpareICE/avhrr_345_mhs_345_angles_tsurfcfsr_all_global_h/2010/02/"
# sfiles += glob.glob(os.path.join(spath + "*/*.gz"))

# spath = "/home/inderpreet/Dendrite/SatData/SpareICE/avhrr_345_mhs_345_angles_tsurfcfsr_all_global_h/2009/12/"
# sfiles += glob.glob(os.path.join(spath + "*/*.gz"))

siwp = spareICE(sfiles)

mask = np.abs(siwp.lat) <= 65 


with open(outfile + ".pickle", "wb") as f:
    pickle.dump(siwp.lat[mask], f)
    pickle.dump(siwp.lon[mask], f)
    pickle.dump(siwp.iwp[mask], f)

f.close()