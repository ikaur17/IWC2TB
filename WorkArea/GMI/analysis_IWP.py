#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 20:35:20 2021

@author: inderpreet
"""


import numpy as np
import pickle
import xarray
from iwc2tb.GMI.grid_field import grid_field
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib import cm
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 16})



#%%
def plot_selective_dardar(tlat, tlon, tiwp, latlims, lonlims):


    lamask = np.logical_and(tlat > latlims[0], tlat < latlims[1])
    lomask = np.logical_and(tlon > lonlims[0], tlon < lonlims[1])

    tmask  = np.logical_and(lamask, lomask)
    fig, ax = plt.subplots(1, 1, figsize = [20, 10])

    m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65, ax = ax)
    m.drawcoastlines()

    cs = m.scatter(tlon[tmask], tlat[tmask],  c =  tiwp[tmask],
                   norm=colors.LogNorm(vmin=1e-3, vmax= 50),
                   cmap = cm.rainbow)
    fig.colorbar(cs)
    parallels = np.arange(-80.,80,20.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True,False,True,False])
    meridians = np.arange(0.,360.,40.)
    m.drawmeridians(meridians,labels=[True,False,False,True])

def zonal_mean(lat, iwp, latbins):


    bins     = np.digitize(lat, latbins)

    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)

    return iwp_mean, nbins


def histogram(iwp, bins):

    hist, _ = np.histogram(iwp, bins)

    return hist/np.sum(hist)


def plot_hist(siwp, diwp, giwp, giwp0, tiwp, slat, dlat, glat, tlat, latlims, key = 'all'):


    bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,.5,1,2, 5, 10, 20, 50, 100, 200, 1000])

    smask = np.logical_and(np.abs(slat) >= latlims[0] , np.abs(slat) <= latlims[1])
    dmask = np.logical_and(np.abs(dlat) >= latlims[0] , np.abs(dlat) <= latlims[1])
    gmask = np.logical_and(np.abs(glat) >= latlims[0] , np.abs(glat) <= latlims[1])
    tmask = np.logical_and(np.abs(tlat) >= latlims[0] , np.abs(tlat) <= latlims[1])

    shist = histogram(0.001 * siwp[smask], bins)

    dhist = histogram(diwp[dmask], bins)
    ghist = histogram(giwp[gmask], bins)
    thist = histogram(tiwp[tmask], bins)
    ghist0 = histogram(giwp0[gmask], bins)

    bin_center = 0.5 * (bins[1:] + bins[:-1])
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])

    ax.plot(bin_center, shist, 'b-.', markersize = 2, label = "SI" )
    ax.plot(bin_center, dhist, 'b--', markersize = 2, label = "DARDAR" )
    #ax.plot(bin_center, thist, 'o-', markersize = 2,  label = "DARDAR training" )
    ax.plot(bin_center, ghist, 'b-', markersize = 2, label = "GMI QRNN" )
    ax.plot(bin_center, ghist0, 'r:', markersize = 2, label = "GMI GPROF" )


    ax.set_xlabel("IWP [kg/m2]")
    ax.set_ylabel("frequency")
    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title(str(latlims[0]) +"-" +str(latlims[1]))
    fig.savefig("Figures/PDF_IWP_" +str(latlims[0]) +"-" +str(latlims[1]) + key + ".png", bbox_inches = "tight")

#%% spare ice data
with open("spareice_jan2020.pickle", "rb") as f:
    slat = pickle.load(f)
    slon = pickle.load(f)
    siwp = pickle.load(f)
f.close()
smask = np.abs(slat) <= 65.0
siwpg, siwpc =  grid_field(slat[smask], slon[smask]%360, siwp[smask],
                                  gsize = 2.0, startlat = 65.0)

with open("gridded_spareice_jan.pickle", "wb") as f:
    pickle.dump(siwpg, f)
    pickle.dump(siwpc, f)
    f.close()

#%% dardar data
with open("dardar_jan2017.pickle", "rb") as f:
    dlat = pickle.load(f)
    dlon = pickle.load(f)
    diwp = pickle.load(f)
f.close()

dmask = np.abs(dlat) <= 65.0
diwpg, diwpc =  grid_field(dlat[dmask], dlon[dmask]%360, diwp[dmask],
                                  gsize = 5, startlat = 65.0)

with open("gridded_dardar_2017.pickle", "wb") as f:
    pickle.dump(diwpg, f)
    pickle.dump(diwpc, f)
    f.close()

#%%
with open("dardar_jan2009.pickle", "rb") as f:
    dlat1 = pickle.load(f)
    dlon1 = pickle.load(f)
    diwp1 = pickle.load(f)
    f.close()

    dmask = np.abs(dlat1) <= 65.0
    diwpg1, diwpc1 =  grid_field(dlat1[dmask], dlon1[dmask]%360, diwp1[dmask],
                                      gsize = 5, startlat = 65.0)

with open("gridded_dardar_2009.pickle", "wb") as f:
    pickle.dump(diwpg1, f)
    pickle.dump(diwpc1, f)
    f.close()

#%% read compiled IWP for one month
# janfile = os.path.expanduser("jan2020_IWP_sgd_old.pickle")
# with open(janfile, "rb") as f:
#       giwp  = pickle.load(f)
#       giwp_mean  = pickle.load(f)
#       giwp0 = pickle.load(f)
#       glon  = pickle.load(f)
#       glat  = pickle.load(f)
#       glsm  = pickle.load(f)

#       f.close()

# a = giwp.copy()
# a[:731740, : ] = giwp0
# giwp0 = a.copy()

#giwp_mean = np.concatenate(giwp_mean, axis = 0)

# giwp[giwp < 1e-4] = 0
# giwp_mean[giwp_mean < 1e-4] = 0

# giwp_mean[np.abs(glat)> 65.0] = np.nan
# giwp[np.abs(glat)> 65.0] = np.nan

# # giwp[giwp > giwp_mean]  = np.nan
# # giwp_mean[giwp > giwp_mean]  = np.nan
inputfile = "jan2019_IWP_lpa1.nc"

dataset1 = xarray.open_dataset(inputfile)

# #giwp = dataset.IWP.data
# giwp_mean = dataset.iwp_mean.data
# giwp0 = dataset.iwp0.data
# glon = dataset.lon.data
# glat = dataset.lat.data
# glsm = dataset.lsm.data

# #giwp[giwp < 1e-4] = 0
# #giwp_mean[giwp_mean < 1e-4] = 0
# #giwp[giwp > giwp_mean]  = np.nan
# #giwp_mean[giwp > giwp_mean]  = np.nan
# dataset.close()

#%%

inputfile = "jan2020_IWP_lpa1.nc"

dataset = xarray.open_dataset(inputfile)

#dataset = xarray.combine_nested([dataset1, dataset2], concat_dim=["pixels"])

#giwp = dataset.IWP.data
giwp_mean = dataset.iwp_mean.data
giwp0 = dataset.iwp0.data
glon = dataset.lon.data
glat = dataset.lat.data
glsm = dataset.lsm.data

#giwp[giwp < 1e-4] = 0
#giwp_mean[giwp_mean < 1e-4] = 0
#giwp[giwp > giwp_mean]  = np.nan
#giwp_mean[giwp > giwp_mean]  = np.nan
#dataset1.close()
#dataset2.close()



#%% grid IWP

#giwp[giwp < 0] = 0
nanmask = ~np.isnan(giwp_mean)
gmask = np.abs(glat) <= 65.0

gmask = np.logical_and(gmask, nanmask)
#gmask[:100000, :] = False
giwpg, giwpc =  grid_field(glat[gmask].ravel(), glon[gmask].ravel()%360,
                           giwp_mean[gmask].ravel(),
                           gsize = 2, startlat = 65.0)

#%% grid IWP0
nanmask = giwp0 > -9000
gmask = np.abs(glat) <= 65.0

gmask = np.logical_and(gmask, nanmask)
#gmask[:100000, :] = False
giwp0g, giwp0c =  grid_field(glat[gmask].ravel(), glon[gmask].ravel()%360,
                           giwp0[gmask].ravel(),
                           gsize = 2, startlat = 65.0)


#%%
with open("gridded_iwp_lpa_jan2017_v.pickle", "wb") as f:

    pickle.dump(giwpg, f)
    pickle.dump(giwpc, f)
    pickle.dump(giwp0g, f)
    pickle.dump(giwp0c, f)

    f.close()

#%%
bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,
                 .5,1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 50])

bin_center = 0.5 * (bins[1:] + bins[:-1])

ghist = histogram(giwp_mean.ravel(), bins)
shist = histogram(siwp * 0.001, bins)
dhist = histogram(diwp, bins)


fig, ax = plt.subplots(1, 1, figsize = [8, 8])
ax.plot( bin_center, ghist, label = "QRNN")
ax.plot( bin_center, shist, label = "spareice")
ax.plot( bin_center,dhist,  label = "dardar")
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend()
ax.set_ylabel("PDF")
ax.set_xlabel("IWP")
fig.savefig("PDF_IWP_GMI.pdf", bbox_inches  = "tight")

#%% plot histograms, all

giwp = giwp_mean
lsmmask = np.ones(giwp.shape, dtype = "bool")

#lsmmask = glsm == 0

#tmask = train.ta.stype == 0

latlims = [0, 30]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat,
          dlat, glat[lsmmask], dlat, latlims )

latlims = [30, 45]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims )

latlims = [45, 65]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims )


#%% plot histograms, water
lsmmask = np.ones(giwp.shape, dtype = "bool")

lsmmask = glsm == 0
#tmask = train.ta.stype == 0

latlims = [0, 30]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat,
          dlat, glat[lsmmask], dlat, latlims, key = "water" )

latlims = [30, 45]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims , key = "water")

latlims = [45, 65]
plot_hist(siwp, diwp, giwp[lsmmask], giwp0[lsmmask], diwp, slat, dlat,
          glat[lsmmask], dlat, latlims, key = "water" )





#%% plot zonal_means
plt.rcParams.update({'font.size': 18})
latbins = np.arange(-66, 65, 3.0)

giwp = giwp_mean
nanmask = giwp0 > -9000

ziwp0, ziwp0c = zonal_mean(glat[nanmask], giwp0[nanmask], latbins)

nanmask = ~np.isnan(giwp)
nanmask = ~(np.isnan(giwp))
#nanmask = ~nanmask & (glsm == 0)
ziwp, ziwpc = zonal_mean(glat[nanmask], giwp[nanmask], latbins)


ziwp_si, ziwp_sic = zonal_mean(slat, siwp, latbins)
ziwp_d, ziwp_dc = zonal_mean(dlat,  diwp, latbins)
#ziwp_d1, ziwp_dc1 = zonal_mean(dlat1,  diwp1, latbins)


fig, ax = plt.subplots(1, 1, figsize = [12, 12])
ax.plot(ziwp[:-1]/ziwpc[:-1],latbins, 'r-',  label = r"QRNN (99.0 kg m$^{-2}$)")
#ax.plot(ziwp/ziwpc,latbins[:-1], 'b-',  label = "QRNN")


#ax.plot(ziwp/ziwpc,latbins, 'b-',  label = "QRNN")
ax.plot(ziwp0[:-1]/ziwp0c[:-1], latbins, 'b.-', label = "GPROF (26.4 kg m$^{-2}$)")
#ax.plot(ziwp0/ziwp0c, latbins, 'b.-', label = "GPROF")

ax.plot( (0.001 * ziwp_si[:-1]/ziwp_sic[:-1]), latbins, 'b--', label = "SpareIce (92.7 kg m$^{-2}$)")
ax.plot(ziwp_d[:-1]/ziwp_dc[:-1],latbins, 'b:', label = "DARDAR-17 (123.4 kg m$^{-2}$)")

#ax.plot(ziwp_d1[:-1]/ziwp_dc1[:-1],latbins, 'k', label = "DARDAR-09 (131.5 kg m$^{-2}$)")



ax.set_ylabel("Latitude [deg]")
ax.set_xlabel("IWP [kg/m2]")
ax.legend()
ax.grid("on", alpha = 0.3)
fig.savefig("Figures/zonal_mean_all_jan.pdf", bbox_inches = "tight")

#%% plot zonal_means from gridded data

lats = np.arange(-65, 65, 2.0)
lats1 = np.arange(-65, 65, 5.0)
gziwp_s = np.mean(siwpg/siwpc, axis = 1)
gziwp_d1 = np.mean(diwpg1/diwpc1, axis = 1)
gziwp_g = np.mean(giwpg/giwpc, axis = 1)
gziwp_g0 = np.mean(giwp0g/giwp0c, axis = 1)


fig, ax = plt.subplots(1, 1, figsize = [15, 15])

ax.plot(0.001 * gziwp_s, lats, label = "SI")
ax.plot(gziwp_d1, lats1, label = "DARDAR")
ax.plot(gziwp_g, lats, label = "QRNN")
ax.plot(gziwp_g0, lats, label = "GPROF")
ax.legend()

ax.set_ylabel("Latitude [deg]")
ax.set_xlabel("IWP [kg/m2]")
ax.grid("on", alpha = 0.3)

fig.savefig("Figures/zonal_mean_gridded.png", bbox_inches = "tight")


#%% get avg IWP, weighted by cosine of latitude [g/m2]

lats  = np.arange(-65, 65, 2)
cosines = np.cos(np.deg2rad(lats))

lats1  = np.arange(-65, 65, 5)
cosines1 = np.cos(np.deg2rad(lats1))

print ("SI mean: ", np.sum(gziwp_s * cosines)/np.sum(cosines))
print ("DARDAR mean: ", 1000 * np.sum(gziwp_d1 * cosines1)/np.sum(cosines1))
print ("QRNN mean: ", 1000 * np.sum(gziwp_g * cosines)/np.sum(cosines)) # g/m2

im = ~np.isnan(gziwp_g0)
print ("GPROF mean: ", 1000 * np.sum(gziwp_g0[im] * cosines[im])/np.sum(cosines))

#%% spatial distribution
lon = np.arange(0, 360, 2)
lat = np.arange(-65, 65, 2)
lon1 = np.arange(0, 360, 5)
lat1 = np.arange(-65, 65, 5)


fig, axes = plt.subplots(2, 2, figsize = [30, 10])

ax = axes.ravel()
m = Basemap(projection= "cyl", llcrnrlon = 0,  llcrnrlat = -65, urcrnrlon = 360,
            urcrnrlat = 65, ax = ax[3], fix_aspect = False, anchor ="C")
m.drawcoastlines()
m.pcolormesh(lon, lat, 0.001 * siwpg/siwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 4),
             cmap = cm.jet)

parallels = np.arange(-80.,80,20.)
meridians = np.arange(0.,360.,40.)
# labels = [left,right,top,bottom]
m.drawparallels(parallels,labels=[False,False,False,False])

m.drawmeridians(meridians,labels=[False,False,False,True])


m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360,
            urcrnrlat = 65, ax = ax[2], fix_aspect = False)
m.pcolormesh(lon1, lat1, diwpg/diwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 4),
             cmap = cm.jet)
m.drawcoastlines()
m.drawparallels(parallels,labels=[True,False,False,False])

m.drawmeridians(meridians,labels=[True,False,False,True])

m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360, urcrnrlat = 65,
            ax = ax[0], fix_aspect = False)
cs = m.pcolormesh(lon, lat, giwpg/giwpc, norm=colors.LogNorm(vmin=1e-4, vmax= 4),
                  cmap = cm.jet)
m.drawcoastlines()
m.drawparallels(parallels,labels=[True,False,False,False])

m.drawmeridians(meridians,labels=[False,False,False,False])

m = Basemap(projection= "cyl", llcrnrlon = 0, llcrnrlat = -65, urcrnrlon = 360,
            urcrnrlat = 65, ax = ax[1], fix_aspect = False)
cs = m.pcolormesh(lon, lat, giwp0g/giwp0c, norm=colors.LogNorm(vmin=1e-4, vmax= 4),
                  cmap = cm.jet)
m.drawcoastlines()
m.drawparallels(parallels,labels=[False,False,False,False])

m.drawmeridians(meridians,labels=[False,False,False,False])
fig.colorbar(cs, label=r"IWP [kg m$^{-2}$]", ax = axes, shrink = 0.8)
#plt.tight_layout(pad = 6.0)

ax[3].set_title("SpareICE")
ax[2].set_title("DARDAR")
ax[0].set_title("Q IWP")
ax[1].set_title("GPROF")

ax[0].text(0.5, 66, "a)")
ax[1].text(0.5, 66, "b)")
ax[2].text(0.5, 66, "c)")
ax[3].text(0.5, 66, "d)")
fig.savefig("Figures/IWP_spatial_distribution.pdf", bbox_inches = "tight")
