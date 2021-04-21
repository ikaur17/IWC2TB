#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:33:05 202

@author: inderpreet
"""
import numpy as np
from typhon.arts import xml
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from matplotlib import cm
from array import array
plt.rcParams.update({'font.size': 20})
from era2dardar.utils.alt2pressure import pres2alt, alt2pres
from era2dardar.DARDAR import DARDARProduct
from era2dardar.utils.Z2dbZ import Z2dbZ
from scipy import interpolate
import os
#from iwc2tb.GMI.L2CSNOW import SNOW2C
from pansat.products.satellite.cloud_sat import l2b_geoprof

#%%

option1 = True
option2 = False

if option1:
    dpath  =  '/home/inderpreet/Dendrite/UserAreas/Kaur//Z'
        
    dpath_srtm  =  '/home/inderpreet/Dendrite/UserAreas/Kaur//Z'

    bpfile  = os.path.join(dpath, "particle_bulkprop_field.xml.bin")    
    bpfile_s  = os.path.join(dpath_srtm, "particle_bulkprop_field.xml.bin")
    
    latfile = os.path.join(dpath, 'lat_true.xml.bin')

    pfile  = os.path.join(dpath, 'p_grid.xml.bin')
    zfile   = os.path.join(dpath, 'reflectivities.xml.bin')
    zfieldfile = os.path.join(dpath, 'z_field.xml.bin')
    zsurface  = os.path.join(dpath, 'z_surface.xml.bin')
    zsurface_s  = os.path.join(dpath_srtm, 'z_surface.xml.bin')
    
    path = "/home/inderpreet/Downloads" 
    file = "2010027071721_19950_CS_2B-GEOPROF_GRANULE_P1_R05_E03_F00.hdf"    
    path1 = "/home/inderpreet/Dendrite/Projects/IWP/GMI/2C-SNOW/"
    file1 = "2010027071721_19950_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E03_F00.hdf"
    ix, iy = 84, 13633
    #ix, iy = 84, 6827
    #ix, iy = 84, 6810
    #latlims = [35.5, 36]
    latlims = [0, 65]
    #latlims = [-65, 0]
    #latlims = [32, 36]
    lonlims = [70, 100]




bp = np.fromfile(bpfile,  dtype = np.dtype('d'))
bp = bp.reshape(2, ix, iy)

bp = np.where(bp == 0, np.nan, bp)

bp_s = np.fromfile(bpfile_s,  dtype = np.dtype('d'))
bp_s = bp_s.reshape(2, ix, iy)

bp_s = np.where(bp_s == 0, np.nan, bp_s)

Z = np.fromfile(zfile,  dtype = np.dtype('d'))
Z = Z.reshape(ix, iy)

h = np.fromfile(zfieldfile, dtype = np.dtype('d'))
h = h.reshape(ix, iy)

lat = np.fromfile(latfile, dtype = np.dtype('d'))
lat = lat

p = np.fromfile(pfile, dtype = np.dtype('d'))
#h = pres2alt(p)

sur = np.fromfile(zsurface, dtype = np.dtype('d'))
sur_s = np.fromfile(zsurface_s, dtype = np.dtype('d'))

lats = np.repeat(lat.reshape(1, -1), 84, axis = 0)

mask  = (lat >= latlims[0]) & (lat <= latlims[1])


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 22))
Z = np.where(Z < -30, np.nan, Z)

cs= ax1.pcolormesh(lats[:, mask], h[:, mask]/1000, Z[:, mask], norm=Normalize(-30, 40),
               cmap = cm.rainbow)
fig.colorbar(cs, label="dbZe", ax = ax1)

ax1.set_title("Radar Reflectivity", loc="left")
ax1.set (ylim=(-1, 10))
ax1.plot(lat[mask], sur_s[mask]/1000, linewidth = 5, color = 'k')
ax1.plot(lat[mask], sur[mask]/1000, linewidth = 5, color = 'k')
#plt.gca().invert_yaxis()
ax1.set_xlabel("Latitude [deg]")
ax1.set_ylabel("Height [km]");

cs1= ax2.pcolormesh(lats[:, mask], h[:, mask]/1000, bp[1, :, mask].T, norm=Normalize(0, 0.001),
                    cmap = cm.rainbow)
fig.colorbar(cs1, label="IWC[kg/m3]", ax = ax2)

ax2.set_title("IWC", loc="left")
ax2.set (ylim=(-1, 10))
ax2.plot(lat[mask], sur[mask]/1000, linewidth = 5, color = 'k')

#plt.gca().invert_yaxis()
ax2.set_xlabel("Latitude [deg]")
ax2.set_ylabel("Height [km]");

cs1= ax3.pcolormesh(lats[:, mask], h[:, mask]/1000, bp_s[0, :, mask].T, norm=Normalize(0, 0.001),
                    cmap = cm.rainbow)
fig.colorbar(cs1, label="RWC[kg/m3]", ax = ax3)

ax3.set_title("RWC", loc="left")
ax3.set (ylim=(-1, 10))
ax3.plot(lat[mask], sur_s[mask]/1000, linewidth = 5, color = 'k')

#plt.gca().invert_yaxis()
ax3.set_xlabel("Latitude [deg]")
ax3.set_ylabel("Height [km]");

fig.savefig("IWC_RWC.pdf", bbox_inches = "tight")

#%%

fig1, ax = plt.subplots(1, 1, figsize = [8, 12])
ax.plot( bp_s[1, :, mask][0], h[:, mask][:,0], marker = '.', markersize = '7')
ax.plot( bp_s[1, :, mask][0][0], sur_s[mask][0], 'o', markersize = '7')

#ax.set_yticks(h[::3])

ax.plot( bp_s[1, :, mask][10], h[:, mask][:, 10], marker = '.', markersize = '7')
ax.plot( bp_s[1, :, mask][10][0], sur_s[mask][10], 'o', markersize = '7')
ax.set(ylim = [0, 10000])
ax.set_ylabel("height [m]")
ax.set_xlabel("IWC [kg/m3]")
plt.grid()
fig1.savefig("IWP_profiles", bbox_inches = "tight")



#%%



lats = np.repeat(lat.reshape(-1, 1), 84, axis = 1)



fig, ax = plt.subplots(1, 1, figsize = [8, 8])
h1 = h[:, mask]
lats1 = lats[mask, :]
bp1 = bp[1, :, mask]
Z1 = Z[:, mask]
cs = ax.pcolormesh(lats1.T, h1/1000, Z1, norm=Normalize(-30, 30), cmap = cm.rainbow)
fig.colorbar(cs, label="IWC[kg/m3]", ax = ax)
ax.plot(lat[mask], sur_s[mask]/1000, linewidth = 1, color = 'k')




    

      
    
    
