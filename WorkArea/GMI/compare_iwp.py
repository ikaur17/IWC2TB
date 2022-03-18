#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:13:02 2021

@author: inderpreet
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from iwc2tb.GMI.GMI import GMI
import zipfile
import typhon.arts.xml as xml
from era2dardar.utils import zip2dardar
from era2dardar.DARDAR import DARDARProduct
from era2dardar.atmData import atmdata
from era2dardar.utils.alt2pressure import alt2pres, pres2alt
from iwc2tb.common.plot_locations_map import plot_locations_map
from era2dardar.utils.read_from_zip import read_from_zip
import shutil
plt.rcParams.update({'font.size': 18})
from scipy import interpolate

#%%
def bin_iwp(lat, iwp, latbins = None):

    if latbins is None:

        latbins  = np.arange(-65, 66, 2)

    bins     = np.digitize(lat, latbins)

    nbins    = np.bincount(bins)
    iwp_mean = np.bincount(bins, iwp)/nbins

    return iwp_mean, latbins

#%%
def interpolate_iwc(dardar, z_field, p_grid):
        """
        The IWC data from DARDAR interpolated to pressure grid defined in
        p_grid
        -------
        grid_iwc : np.array containing the interpolated values in
        dimensions [1, p, lat, lon]
        """



        try:
            iwc             = dardar.iwc
            height_d        = dardar.height
            lat             = dardar.latitude
        except:
            print ("iwc and height not available as class methods/property")


        z_field         = np.squeeze(z_field)

        grid_iwc = np.zeros(z_field.shape)

        for i in range(lat.shape[0]):
            # first interpolate dardar heights to pressures using z_field and p_grid
            f                 = interpolate.interp1d(z_field[:, i], np.log(p_grid), fill_value  = "extrapolate")
            p_d               = f(height_d) # log scale
            # using dardar pressure levels to interpolate reflectivities to p_grid
            f                 = interpolate.interp1d(p_d, iwc[i, :], fill_value = "extrapolate")
            grid_iwc[:, i]    = f(np.log(p_grid))


        return grid_iwc
#%%
def dardar_iwp(zipfiles):

    p_grid_fine = alt2pres(np.arange(-700, 8000, 125))
    p_grid_coarse = alt2pres(np.arange(8000, 20000, 250))
    p_grid = (np.concatenate([p_grid_fine, p_grid_coarse,
                             np.array([30, 20, 10, 7, 5, 3, 2, 1]) * 100]))

    iwp_total = []
    lat_all = []
    lon_all = []

    for zfile in zipfiles:
        print (zfile)
        dardarfile, N = zip2dardar.zip2dardar(zfile)
        z_field = read_from_zip(zfile, "z_field")

        dardar = DARDARProduct(dardarfile, latlims = [-65, 65], node = N)

        iwc = interpolate_iwc(dardar, z_field, p_grid)

#------------------------------------------------
        z0 = z_field[0, :] + 1200

        for i in range(z_field.shape[1]):
            mask = z_field[:, i] < z0[i]
            iwc[mask, i] = 0

#------------------------------------------------

        lat = dardar.latitude
        lon = dardar.longitude



        z   = pres2alt(p_grid)

        iwp = np.zeros(lat.shape)

        for i in range(iwc.shape[1]):
            iwp[i] = np.trapz(iwc[:, i], z)

        iwp_total.append(iwp)
        lat_all.append(lat)
        lon_all.append(lon)

    return np.concatenate(iwp_total), np.concatenate(lat_all), np.concatenate(lon_all)

#%%
def filter_iwp(latlims, lonlims, lat, lon):

    im1  = (lat > latlims[0]) & (lat < latlims[1])
    im2  = (lon > lonlims[0]) & (lon < lonlims[1])
    mask = np.logical_and(im1, im2)
    return mask

#%%
def get_basenames(files):
    basefiles = []
    for file in files:
        basefiles.append(os.path.basename(file))
    return basefiles

def get_fullnames(path , files):

    fullnames = []
    for file in files:
        fullnames.append(os.path.join(path, file))
    return fullnames
#%%
if __name__ == "__main__":
    # GMI simulations
    #inpath   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/test/test1.3')
    #inpath1  =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_testsimulations/test_f07')
    inpath2  =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1')



    #matfiles = glob.glob(os.path.join(inpath, "2010_*.mat"))
    #matfiles1 = glob.glob(os.path.join(inpath1, "2010_0*.mat"))
    matfiles = glob.glob(os.path.join(inpath2, "2009_00*.mat"))
    matfiles1 = glob.glob(os.path.join(inpath2, "2009_01*.mat"))
    matfiles2 = glob.glob(os.path.join(inpath2, "2009_02*.mat"))
    matfiles3 = glob.glob(os.path.join(inpath2, "2009_03*.mat"))

    matfiles  = matfiles + matfiles1 + matfiles2 + matfiles3
# #%% find files with high IWC

#     #Himalayas
#     latlims = [29, 44]
#     lonlims = [71, 76]

#     # N. America
#     latlims = [34, 40]
#     lonlims = [240, 242]
#     for file in matfiles2:
#             gmi = GMI(file)

#             iwp = gmi.iwp

#             iwp_mean = np.mean(iwp)

#             zfile = gmi.get_inputfiles()
#             diwp, dlat, dlon = dardar_iwp(zfile)

#             diwp_mean = np.mean(diwp)

#             diff = np.abs((diwp_mean - iwp_mean)/diwp_mean)

#             if diff > 0.35:
#                 print(file, diwp_mean, iwp_mean)

#             #mla = (gmi.lat > latlims[0]) & (gmi.lat < latlims[1])
#             #mlo = (gmi.lon > lonlims[0]) & (gmi.lon < lonlims[1])

#             #mask = np.logical_and(mla, mlo)

#             #if np.sum(mask) != 0:
#             #    print (file)
#             #    plot_locations_map(gmi.lat, gmi.lon, gmi.iwp)


    matfiles = matfiles[:100]

#%%
    gmi = GMI(matfiles)
    glat = gmi.lat.ravel()
    glon = gmi.lon.ravel()%360
    giwp = gmi.iwp.ravel()

    gmask = np.abs(glat <= 65.0)

#%%

    zipfiles = gmi.get_inputfiles()

    diwp, dlat, dlon = dardar_iwp(zipfiles)
    dlon = dlon%360

    dmask = np.abs(dlat <= 65.0)

    stype  = gmi.stype


#%%
    iwp_mean_gmi, latbins = bin_iwp(glat, giwp)
    iwp_mean_dardar, latbins = bin_iwp(dlat, diwp)


#%% bootstrap error for simulations
    N = 500
    iwp_mean_bs = np.zeros([N, iwp_mean_gmi.size])
    for j in range(N):
        iargs = np.random.randint(0, giwp.size, int(giwp.size*0.1))
        iwp_mean_bs[j, :], _ = bin_iwp(glat[iargs], giwp[iargs])


    std_iwp = np.std(iwp_mean_bs, axis = 0)

#%%
    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    ax.plot(iwp_mean_gmi[:], latbins, 'b--', label = "LPA + F07T")
    ax.plot(iwp_mean_dardar[:], latbins, 'b', label = "DARDAR")


    ax.set_xlabel(r"IWP [kg m$^{-2}$]")
    ax.set_ylabel(r"Latitude [$\circ$]")
    ax.grid("on", alpha = 0.3)
    ax.legend()
    fig.savefig("Figures/IWP_GMI_dardar.png", bbox_inches = "tight")


#%%
    dardar_error = 0.4 * iwp_mean_dardar

#%% PDF of IWP

    fig, ax = plt.subplots(1, 2, figsize = [16, 10])
    ax = ax.ravel()
    ax[0].plot(iwp_mean_dardar[:], latbins, c = "tab:blue", linewidth = 2.0, label = "DARDAR")
    ax[0].plot(iwp_mean_gmi[:], latbins, c = "tab:red", linewidth = 2.0, label = "LPA + F07T")

    ax[0].fill_betweenx(latbins, iwp_mean_dardar[:]-dardar_error,
                        iwp_mean_dardar[:]+dardar_error,
                        color = "tab:blue", alpha = 0.2)

    ax[0].fill_betweenx(latbins, iwp_mean_gmi[:] - std_iwp,
                            iwp_mean_gmi[:] + std_iwp,
                            color = "tab:red", alpha = 0.2)




    ax[0].set_xlabel(r"IWP [kg m$^{-2}$]")
    ax[0].set_ylabel(r"Latitude [deg]")
    ax[0].grid("on", alpha = 0.3)
    ax[0].legend()
    #fig.savefig("Figures/IWP_GMI_dardar.png", bbox_inches = "tight")





    bins = np.array([0.0,.0001,.00025,.0005,0.001,.0025,.005,.01,.025,.05,.1,.25,.5,1,2, 5, 10, 15, 20, 25])

    #bins = np.arange(0, 200, 0.0001)
    ghist, _ = np.histogram(giwp[gmask], bins,  density = True )
    dhist, _ = np.histogram(diwp[dmask], bins,  density = True)



    bin_center = 0.5 * (bins[1:] + bins[:-1])

    ax[1].plot(bin_center, dhist, 'o-', color = "tab:blue", label = "DARDAR", linewidth = 2.0 )
    ax[1].plot(bin_center, ghist, 'o-', color = "tab:red", label = "LPA + f07t", linewidth = 2.0 )

    ax[1].set_xlabel(r"IWP [kg m$^{-2}$]")
    ax[1].set_ylabel("PDF")
    #ax.legend()
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    ax[1].grid("on", alpha = 0.3)
    ax[0].set_xlim([0, 0.35 ])
    ax[0].text(0.005, 73, "a)")
    ax[1].text(4.08e-5, 21990, "b)")

    fig.savefig("Figures/PDF_IWP_DARDAR.pdf", bbox_inches = "tight")


#%% pratio PDF

    pr = gmi.get_O("pratio_gmi")
    pr = np.stack(pr).ravel()

    fig, ax = plt.subplots(1, 1, figsize = [8, 8])
    ax.hist(pr, bins = np.arange(1, 1.41, 0.01), density = True, histtype = "step")
    ax.set_xlabel("pratio")
    ax.set_ylabel("frequency")
    ax.set_title("mean pratio = " + str(np.round(np.mean(pr), 3)))
    fig.savefig("Figures/pratio.png", bbox_inches = "tight")


#%%
