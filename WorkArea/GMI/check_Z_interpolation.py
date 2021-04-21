#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:16:26 2021

@author: inderpreet
"""


from era2dardar.utils.read_from_zip import read_from_zip
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from matplotlib import cm
from era2dardar.utils.Z2dbZ import Z2dbZ

path_new = os.path.expanduser("~/Dendrite/Projects/IWP/GMI/DARDAR_ERA_m65_p65_SRTM")
path_old = os.path.expanduser("~/Dendrite/Projects/IWP/GMI/DARDAR_ERA_m65_p65_zfield/srtm/")
file = "2010_027_07_A.zip"
zfile_old = os.path.join(path_old, file)
zfile_new = os.path.join(path_new, file)


parameter = "reflectivities"

Z_old = read_from_zip(zfile_old, "reflectivities")
sur_old = read_from_zip(zfile_old, "z_surface")
z_field = read_from_zip(zfile_old, "z_field")

Z_new = read_from_zip(zfile_new, "reflectivities")
sur_new = read_from_zip(zfile_new, "z_surface")
lat     = read_from_zip(zfile_new, "lat_grid")

lats = np.repeat(lat.reshape(1, -1), 91, axis = 0)

mask = (lat > 35.5) & (lat < 36)

Z_new = Z2dbZ(Z_new)
Z_old = Z2dbZ(Z_old)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = [20, 10])
cs = ax1.pcolormesh(lats[:, mask], z_field[:, mask], Z_new[:, mask], norm=Normalize(-30, 20), cmap = cm.rainbow)
ax2.pcolormesh(lats[:, mask], z_field[:, mask], Z_old[:, mask], norm=Normalize(-30, 20), cmap = cm.rainbow)
fig.colorbar(cs, label="dBZe", ax = (ax1, ax2))
ax1.plot(lat[mask], sur_new[mask], linewidth = 1, color = 'k')
ax2.plot(lat[mask], sur_new[mask], linewidth = 1, color = 'k')

ax1.set(ylim = [0, 10000])
ax2.set(ylim = [0, 10000])

ax1.title('new')
ax2.title('old')
