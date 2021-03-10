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
from iwc2tb.GMI.L2CSNOW import SNOW2C
from pansat.products.satellite.cloud_sat import l2b_geoprof

#%%

option1 = True
option2 = True



if option2:
    dpath_m = "/home/inderpreet/Dendrite/UserAreas/Kaur/temp/f07m/"
    dpath = "/home/inderpreet/Dendrite/UserAreas/Kaur/temp/f07t"
    vmrfile = os.path.join(dpath_m, "vmr_field.xml.bin")
    #filename = '/home/inderpreet/Dendrite/UserAreas/Kaur/particle_bulkprop_field_ascii.xml'
    bpfile    = os.path.join(dpath, "particle_bulkprop_field.xml.bin")
    bpfile_m  = os.path.join(dpath_m, "particle_bulkprop_field.xml.bin")    
    latfile = os.path.join(dpath, 'lat_true.xml.bin')
    pfile  = os.path.join(dpath, 'p_grid.xml.bin')
    zfile   = os.path.join(dpath, 'reflectivities.xml.bin')
    #zfieldfile = '/home/inderpreet/Dendrite/UserAreas/Kaur/2006_2csnow/z_field.xml.bin'
    zsurface  = os.path.join(dpath, 'z_surface.xml.bin')
    path = "/home/inderpreet/Downloads" 
    file = "2006355074548_03451_CS_2B-GEOPROF_GRANULE_P1_R05_E02_F00.hdf"
    path1 = "/home/inderpreet/Dendrite/Projects/IWP/GMI/2C-SNOW/"
    file1 = "2006355074548_03451_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf"    
    ix, iy = 84, 6811
    latlims = [41, 44]
    lonlims = [200, 280]




bp = np.fromfile(bpfile,  dtype = np.dtype('d'))
bp = bp.reshape(2, ix, iy)

bp = np.where(bp == 0, np.nan, bp)

bp_m = np.fromfile(bpfile_m,  dtype = np.dtype('d'))
bp_m = bp_m.reshape(2, ix, iy)

bp_m = np.where(bp_m == 0, np.nan, bp_m)

Z = np.fromfile(zfile,  dtype = np.dtype('d'))
Z = Z.reshape(ix, iy)

#h = np.fromfile(zfieldfile, dtype = np.dtype('d'))
#h = h.reshape(ix, iy)

lat = np.fromfile(latfile, dtype = np.dtype('d'))
lat = lat

p = np.fromfile(pfile, dtype = np.dtype('d'))
h = pres2alt(p)

sur = np.fromfile(zsurface, dtype = np.dtype('d'))

vmr = np.fromfile(vmrfile, dtype = np.dtype('d'))
vmr = vmr.reshape(5, ix, iy)

mask  = (lat >= latlims[0]) & (lat <= latlims[1])


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 22.5))
Z = np.where(Z < -30, np.nan, Z)

cs= ax1.pcolormesh(lat[mask], h/1000, Z[:, mask], norm=Normalize(-30, 40),
               cmap = cm.rainbow)
fig.colorbar(cs, label="dbZe", ax = ax1)

ax1.set_title("Radar Reflectivity", loc="left")
ax1.set (ylim=(0, 10))
ax1.plot(lat[mask], sur[mask]/1000, linewidth = 5, color = 'k')

#plt.gca().invert_yaxis()
#ax1.set_xlabel("Latitude [deg]")
ax1.set_ylabel("Height [km]");

cs1= ax2.pcolormesh(lat[mask], h/1000, bp[1, :, mask].T, norm=Normalize(0, 0.001),
                    cmap = cm.rainbow)
fig.colorbar(cs1, label="IWC[kg/m3]", ax = ax2)

ax2.set_title("IWC f07t", loc="left")
ax2.set (ylim=(0, 10))
ax2.plot(lat[mask], sur[mask]/1000, linewidth = 5, color = 'k')

#plt.gca().invert_yaxis()
#ax2.set_xlabel("Latitude [deg]")
ax2.set_ylabel("Height [km]");

cs1= ax3.pcolormesh(lat[mask], h/1000, bp_m[1, :, mask].T, norm=Normalize(0, 0.001),
                    cmap = cm.rainbow)
fig.colorbar(cs1, label="IWC[kg/m3]", ax = ax3)

ax3.set_title("IWC f07m", loc="left")
ax3.set (ylim=(0, 10))
ax3.plot(lat[mask], sur[mask]/1000, linewidth = 5, color = 'k')

#plt.gca().invert_yaxis()
#ax3.set_xlabel("Latitude [deg]")
ax3.set_ylabel("Height [km]");

#%%

fig1, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.plot( bp[1, :, mask][100], h, marker = '.', markersize = '5')
ax.plot( bp[1, :, mask][100][0], sur[mask][0], 'ro', markersize = '5')

ax.plot( bp[1, :, mask][-1], h, marker = '.', markersize = '5')
ax.plot( bp[1, :, mask][-1][0], sur[mask][0], 'ro', markersize = '5')

ax.set_yticks(h[::3])

ax.plot( bp[1, :, mask][200], h, marker = '.', markersize = '5')
ax.plot( bp[1, :, mask][200][0], sur[mask][10], 'ro', markersize = '5')

ax.plot( bp[1, :, mask][300], h, marker = '.', markersize = '5')
ax.plot( bp[1, :, mask][300][0], sur[mask][10], 'ro', markersize = '5')

ax.set(ylim = [0, 10000])
ax.set_ylabel("height [m]")
ax.set_xlabel("IWC [kg/m3]")
plt.grid()
fig1.savefig("IWP_profiles.pdf", bbox_inches = "tight")





#%%


granule = file[14:19]

file2 = "2010027071721_19950_CS_2C-PRECIP-COLUMN_GRANULE_P1_R05_E03_F00.hdf"


filename = os.path.join(path, file)
filename1 = os.path.join(path1, file1)   
filename2 = os.path.join(path, file2)

s2c = SNOW2C(filename1)
sds_obj = s2c.file.select("snow_water_content") # select sds
swc = sds_obj.get() 

dataset = l2b_geoprof.open(filename)

start = 10000
end = 25000
stride = 100
z = dataset["radar_reflectivity"]/100
z = np.where(z < -30, np.nan, z)
lats = dataset["latitude"]
bins = dataset["bins"]
lons = dataset["longitude"]%360
height = dataset["height"]

#%% interpolate to common height

h   = np.arange(0, 10000, 240)   
grid_z = np.zeros([lats.shape[0], h.shape[0]])
for i in range(lats.shape[0]): 
    f               = (interpolate.interp1d(height[i, :], z[i, :],
                                       fill_value = "extrapolate"))
    grid_z[i, :]          = f(h) 

#%%

mask1  =  (lats >= latlims[0]) & (lats <= latlims[1])
mask2  = (lons <= lonlims[1]) & (lons >= lonlims[0])
mask1 = np.logical_and(mask1, mask2)

#fig, ax = plt.subplots(1, 1, figsize=(20, 6))
# cs = ax.pcolormesh(lats[mask], h/1000, grid_z[mask, :].T, norm=Normalize(-30, 40), cmap = cm.rainbow)
# fig.colorbar(cs, label="dbZe")
# ax.set (ylim=(0, 10))
# #plt.gca().invert_yaxis()
# plt.xlabel(r"Latitude [deg]")
# plt.ylabel("Height [km]");
# plt.title("L2B-GEOPROF", loc="left")
# fig.savefig("cloudsat_radar_reflectivity.pdf", bbox_inches = "tight")

swc1 = np.where(swc < 0, np.nan, swc)
#fig, ax = plt.subplots(1, 1, figsize=(20, 6))
cs2 = ax4.pcolormesh(lats[mask1], height[1, :]/1000, swc1[mask1, :].T /1000,
                     norm=Normalize(0, 0.001), cmap = cm.rainbow)
fig.colorbar(cs2, label="SWC [kg/m3]", ax = ax4)
ax4.set (ylim=(0, 10))
#plt.gca().invert_yaxis()
ax4.set_xlabel(r"Latitude [deg]")
ax4.set_ylabel("Height [km]")
ax4.plot(lat[mask], sur[mask]/1000, linewidth = 5, color = 'k')
ax4.set_title("L2C-SNOW", loc="left")
#fig.savefig("cloudsat_radar_reflectivity.pdf", bbox_inches = "tight"

fig.savefig(f"granule_{granule}_SWC.pdf", bbox_inches = "tight")
#%%
fig, ax = plt.subplots(1, 1, figsize = [20, 8])
vmr = np.where(vmr ==0, np.nan, vmr)
cs1= ax.pcolormesh(lat[mask], h/1000, vmr[4, :, mask].T, 
                    cmap = cm.rainbow)
fig.colorbar(cs1, label="LWC[kg/m3]", ax = ax)

ax.set_title("LWC f07m", loc="left")
ax.set (ylim=(0, 10))
ax.plot(lat[mask], sur[mask]/1000, linewidth = 5, color = 'k')
fig.savefig("LWC.pdf", bbox_inches = "tight")  
    
    
