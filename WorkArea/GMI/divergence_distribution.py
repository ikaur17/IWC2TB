#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:15:34 2021

@author: inderpreet
"""

import numpy as np
import os
import glob
from iwc2tb.GMI.GMI_old import GMI_old
from iwc2tb.GMI.GMI import GMI
from iwc2tb.GMI.GMI_SatData import GMI_Sat
from remove_oversampling_gmi import remove_oversampling_gmi
import random
import scipy
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from iwc2tb.GMI.three_sigma_rule import three_sigma
from scipy.ndimage.filters import gaussian_filter
plt.rcParams.update({'font.size': 22})

#%%
def get_hist2d(ta, tb_gmi):

    # ybins1  = np.arange(-5, 5, 0.5)
    # ybins2 = np.arange(5, 20, 0.75)
    # ybins3 = np.arange(20, 30, 1.25)
    # ybins4 = np.arange(30, 60, 2)
    # ybins = np.concatenate([ybins1, ybins2, ybins3, ybins4])
    counts, _, _  = np.histogram2d(ta[:, 0], ta[:, 0] - ta[:, 1],  
                                        bins=(xbins, ybins))
    
    counts1, _, _ = np.histogram2d(tb_gmi[:, 0].ravel(), 
                                         tb_gmi[:, 0].ravel() - tb_gmi[:,  1].ravel(),  
                                        bins=(xbins, ybins))
    counts[counts == 0] = 1e-5
    counts1[counts1 == 0] = 1e-5

    tbins = xbins.size * ybins.size
    #return counts/tbins, counts1/tbins
    return counts, counts1



#%%
def get_div2d(counts_b, counts_o):
    div = np.sum(np.abs(np.log10(counts_b/counts_o)))/counts_b.size
    
    return div
    
 
    
#%%    
    
inpath_gmi   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2017/01/')
# GMI simulations    
gmifiles = glob.glob(os.path.join(inpath_gmi, "*/*.HDF5"))

random.shuffle(gmifiles)
gmi_sat = GMI_Sat(gmifiles[:20])


key = "lpa"
# GMI frquencies
freq     = ["166.5V", "166.5H",  "183+-7", "183+-3",]   

cols = ["#999999", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


tb_gmi  = gmi_sat.tb
lat_gmi = gmi_sat.lat
lon_gmi = gmi_sat.lon
tb_gmi  = gmi_sat.tb
lsm_gmi = gmi_sat.get_lsm()
iwp_gmi = gmi_sat.iwp


lat_gmi = lat_gmi.ravel()
lon_gmi = lon_gmi.ravel()
tb_gmi  = tb_gmi.reshape(-1, 4)
lsm_gmi = lsm_gmi.ravel()
iwp_gmi = iwp_gmi.ravel()


tb_gmi, lat_gmi, lon_gmi, lsm_gmi, iwp_gmi = remove_oversampling_gmi(tb_gmi, lat_gmi, lon_gmi, lsm_gmi, iwp_gmi)

th = three_sigma(tb_gmi[:, 2])
mask_g = tb_gmi[:, 2] < th

tb_gmi = tb_gmi[mask_g, :]
lsm_gmi = lsm_gmi[mask_g]

#%%
inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/') 
matfiles1 = glob.glob(os.path.join(inpath_mat, "2009_00*.mat"))
matfiles2 = glob.glob(os.path.join(inpath_mat, "2009_01*.mat"))
matfiles3 = glob.glob(os.path.join(inpath_mat, "2009_02*.mat"))
matfiles  = matfiles1 + matfiles2 + matfiles3

gmi                     = GMI(matfiles[:])
ta_aro                  = gmi.ta_noise
stype_aro               = gmi.stype

th = three_sigma(ta_aro[:, 3])
mask_a = ta_aro[:, 3] < th

ta_aro = ta_aro[mask_a, :]
stype_aro = stype_aro[mask_a]


#%%

fig, ax = plt.subplots(1, 1, figsize = [8, 8])

ax.scatter(ta_aro[:, 0], ta_aro[:, 0] - ta_aro[:, 1], s = 5, alpha = 0.8,
           label = "Global simulations", c = "lightsteelblue")#c = pr, vmin = 1, vmax = 1.4 )
ax.set_xlabel("TB 166V GHz [K]" )
ax.set_ylabel("Polarisation difference [K]]")
ax.set_ylim([-5, 20])
ax.set_xlim([100, 280])
ax.grid("on")
fig.savefig("PD_166.png", bbox_inches = "tight")

#%%

inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/lpa_pr_1') 
matfiles1 = glob.glob(os.path.join(inpath_mat, "2009_00*.mat"))
matfiles2 = glob.glob(os.path.join(inpath_mat, "2009_01*.mat"))
matfiles3 = glob.glob(os.path.join(inpath_mat, "2009_02*.mat"))
matfiles  = matfiles1 + matfiles2 + matfiles3
gmi_tro                 = GMI(matfiles[:])
ta_tro                  = gmi_tro.ta_noise
stype_tro               = gmi_tro.stype

th = three_sigma(ta_tro[:, 3])
mask_t = ta_tro[:, 3] < th

ta_tro = ta_tro[mask_t, :]
stype_tro = stype_tro[mask_t]

#%%

inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_testsimulations/test_si') 
matfiles = glob.glob(os.path.join(inpath_mat, "2010_*.mat"))
gmi_tro_si                 = GMI_old(matfiles[:])
ta_tro_si                  = gmi_tro_si.ta_noise
stype_tro_si                  = gmi_tro_si.stype

th = three_sigma(ta_tro_si[:, 3])
mask_s = ta_tro_si[:, 3] < th

ta_tro_si = ta_tro_si[mask_s, :]
stype_tro_si = stype_tro_si[mask_s]

#%%

# inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_testsimulations/test1.2') 
# matfiles = glob.glob(os.path.join(inpath_mat, "2010_*.mat"))
# gmi_tro_12                 = GMI(matfiles[:])
# ta_tro_12                  = gmi_tro_12.ta_noise
# stype_12                   = gmi_tro_12.stype
# # #%%

# inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_testsimulations/test1.3') 
# matfiles = glob.glob(os.path.join(inpath_mat, "2010_*.mat"))
# gmi_tro_13                 = GMI(matfiles[:])
# ta_tro_13                  = gmi_tro_13.ta_noise
# stype_13                   = gmi_tro_13.stype

#%% making subsets with polarisation ratio

mask = (gmi.pratio[mask_a] > 1.07) & (gmi.pratio[mask_a] < 1.12) 
ta_tro_11 = ta_aro[mask]
lsm1 = stype_aro[mask]
mask = (gmi_tro_si.pratio[mask_s] > 1.07) & (gmi_tro_si.pratio[mask_s] < 1.12)  
ta_tro_11_si = ta_tro_si[mask]
lsm2 = stype_tro_si[mask]
ta_tro_11 = np.concatenate([ta_tro_11, ta_tro_11_si], axis = 0)
lsm_11 = np.concatenate([lsm1, lsm2])



mask = (gmi.pratio[mask_a] > 1.27) & (gmi.pratio[mask_a] < 1.33) 
ta_tro_13 = ta_aro[mask]
lsm1 = stype_aro[mask]
mask = (gmi_tro_si.pratio[mask_s] > 1.27) & (gmi_tro_si.pratio[mask_s] < 1.33)  
ta_tro_13_si = ta_tro_si[mask]
lsm2 = stype_tro_si[mask]
ta_tro_13 = np.concatenate([ta_tro_13, ta_tro_13_si], axis = 0)
lsm_13 = np.concatenate([lsm1, lsm2])

mask = (gmi.pratio[mask_a] > 1.17) & (gmi.pratio[mask_a] < 1.23) 
ta_tro_12 = ta_aro[mask]
lsm1 = stype_aro[mask]
mask = (gmi_tro_si.pratio[mask_s] > 1.17) & (gmi_tro_si.pratio[mask_s] < 1.23)  
ta_tro_12_si = ta_tro_si[mask]
lsm2 = stype_tro_si[mask]
ta_tro_12 = np.concatenate([ta_tro_12, ta_tro_12_si], axis = 0)
lsm_12 = np.concatenate([lsm1, lsm2])

mask = (gmi.pratio[mask_a] > 1.35)  
ta_tro_14 = ta_aro[mask]
lsm1 = stype_aro[mask]

mask = (gmi_tro_si.pratio[mask_s] > 1.35)  
ta_tro_14_si = ta_tro_si[mask]
lsm2 = stype_tro_si[mask]
ta_tro_14 = np.concatenate([ta_tro_14, ta_tro_14_si], axis = 0)
lsm_14 = np.concatenate([lsm1, lsm2])
#%% histogram of PDs

# gtbd = tb_gmi[::10, 0] - tb_gmi[::10,  1]
# dtro = ta_tro[:, 0] - ta_tro[:, 1]
# daro = ta_aro[:, 0] - ta_aro[:, 1]

# dtro_12 = ta_tro_12[:, 0] - ta_tro_12[:, 1]
# dtro_13 = ta_tro_13[:, 0] - ta_tro_13[:, 1]
# dtro_14 = ta_tro_14[:, 0] - ta_tro_14[:, 1]

# bins = np.arange(-5, 60, 1.5)
# bin_c = (bins[1:] + bins[:-1])/2

# hist_g, _ = np.histogram(gtbd, bins, density = True)
# hist_t, _ = np.histogram(dtro, bins, density = True)
# hist_a, _ = np.histogram(daro, bins, density = True)
# hist_12, _ = np.histogram(dtro_12, bins, density = True)
# hist_13, _ = np.histogram(dtro_13, bins, density = True)
# hist_14, _ = np.histogram(dtro_14, bins, density = True)

# sigma = 0.1
# hist_g = gaussian_filter(hist_g, sigma)
# hist_t = gaussian_filter(hist_t, sigma)
# hist_a = gaussian_filter(hist_a, sigma)
# hist_12 = gaussian_filter(hist_12, sigma)
# hist_13 = gaussian_filter(hist_13, sigma)
# hist_14 = gaussian_filter(hist_14, sigma)

# fig, ax = plt.subplots(1, 1, figsize = [8, 8])

# ax.plot(bin_c, hist_g, 'k', label = "GMI", linewidth = 2.5)
# ax.plot(bin_c, hist_t, label = "TRO", linewidth = 2.5)            
# ax.plot(bin_c, hist_a, label = "ARO", linewidth = 2.5) 
# ax.plot(bin_c, hist_12, label = "TRO12", linewidth = 2.5)
# ax.plot(bin_c, hist_13, label = "TRO13")
# #ax.plot(bin_c, hist_14, label = "TRO14")


# ax.set_yscale("log")          
         
# ax.legend()
  

#%% make a subset with normal distribution



from scipy.stats import truncnorm
prbins = np.arange(1, 1.45, 0.05)
pr     = gmi.pratio[mask_a]

ndist    = np.random.normal(1.2, 0.24, 100000)

lower, upper = 1, 1.4
mu, sigma = 1.2, 0.24
X = truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

ndist = X.rvs(100000)

nhist, _ = np.histogram(ndist, prbins)

nratio   = nhist/nhist.max()

ipr     = np.digitize(pr, prbins)
icounts = np.bincount(ipr)

ncounts = (icounts[1:] * nratio).astype(int)

ta_normal = []
stype_normal = []


for i in range(8):

         print (i)    

         pr_subset = ta_aro[ipr == i+1, :] 
         stype_subset = stype_aro[ipr == i+1]
        
         args = np.arange(0, ncounts[i], 1)
    
         nargs = np.random.choice(args, size = ncounts[i])
         
         print (nargs.size, ncounts[i], pr_subset.shape)

         ta_normal.append(pr_subset[nargs, :])
         stype_normal.append(stype_subset[nargs])

        
ta_normal = np.concatenate(ta_normal, axis = 0)  
stype_normal = np.concatenate(stype_normal, axis = 0)

#%% calculate divergences for water


xbins = np.arange(80, 280, 5)
ybins = np.arange(-5, 30, 1.5)


xcenter = (xbins[1:] + xbins[:-1])/2
ycenter = (ybins[1:] + ybins[:-1])/2

#%%
# water
fig, ax = plt.subplots(1, 1, figsize = [8, 8])
imask = 0
im = lsm_gmi == 1
im1 = stype_tro == imask
counts_tro, counts_gmi = get_hist2d(ta_tro[im1, :], tb_gmi[im, :])
div_tro = get_div2d(counts_tro, counts_gmi)

im1 = lsm_11 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_11[im1, :], tb_gmi[im, :])
div_11 = get_div2d(counts_tro, counts_gmi)

im1 = lsm_12 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_12[im1, :], tb_gmi[im, :])
div_12 = get_div2d(counts_tro, counts_gmi)


im1 = lsm_13 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_13[im1, :], tb_gmi[im, :])
div_13 = get_div2d(counts_tro, counts_gmi)

im1 = lsm_14 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_14[im1, :], tb_gmi[im, :])
div_14 = get_div2d(counts_tro, counts_gmi)

im1 = stype_aro == imask
counts_aro, counts_gmi = get_hist2d(ta_aro[im1, :], tb_gmi[im, :])
div_aro = get_div2d(counts_aro, counts_gmi)

im1 = stype_normal == imask
counts_aro, counts_gmi = get_hist2d(ta_normal[im1, :], tb_gmi[im, :])
div_nor = get_div2d(counts_aro, counts_gmi)

ax.plot(1, div_tro, 'ko')
ax.plot(1.1, div_11, 'ko')
ax.plot(1.2, div_12, 'ko')
ax.plot(1.3, div_13, 'ko')
ax.plot(1.4, div_14, 'ko')
ax.plot(1.5, div_aro, 'ko')
ax.plot(1.6, div_nor, 'ko')

print (div_tro, div_11, div_12, div_13, div_14, div_aro, div_nor)
#%%
# land
imask = 1
im  = (lsm_gmi >= 3) & (lsm_gmi <= 7) 
im1 = stype_tro == imask
counts_tro, counts_gmi = get_hist2d(ta_tro[im1, :], tb_gmi[im, :])
div_tro = get_div2d(counts_tro, counts_gmi)

im1 = lsm_11 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_11[im1, :], tb_gmi[im, :])
div_11 = get_div2d(counts_tro, counts_gmi)

im1 = lsm_12 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_12[im1, :], tb_gmi[im, :])
div_12 = get_div2d(counts_tro, counts_gmi)


im1 = lsm_13 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_13[im1, :], tb_gmi[im, :])
div_13 = get_div2d(counts_tro, counts_gmi)

im1 = lsm_14 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_14[im1, :], tb_gmi[im, :])
div_14 = get_div2d(counts_tro, counts_gmi)

im1 = stype_aro == imask
counts_aro, counts_gmi = get_hist2d(ta_aro[im1, :], tb_gmi[im, :])
div_aro = get_div2d(counts_aro, counts_gmi)

im1 = stype_normal == imask
counts_aro, counts_gmi = get_hist2d(ta_normal[im1, :], tb_gmi[im, :])
div_nor = get_div2d(counts_aro, counts_gmi)

print (div_tro, div_11, div_12, div_13, div_14, div_aro, div_nor)

ax.plot(1, div_tro, 'ro')
ax.plot(1.1, div_11, 'ro')
ax.plot(1.2, div_12, 'ro')
ax.plot(1.3, div_13, 'ro')
ax.plot(1.4, div_14, 'ro')
ax.plot(1.5, div_aro, 'ro')
ax.plot(1.6, div_nor, 'ro')



#all
counts_tro, counts_gmi = get_hist2d(ta_tro[:, :], tb_gmi[:, :])
div_tro = get_div2d(counts_tro, counts_gmi)

im1 = lsm_11 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_11[:, :], tb_gmi[:, :])
div_11 = get_div2d(counts_tro, counts_gmi)

im1 = lsm_12 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_12[:, :], tb_gmi[:, :])
div_12 = get_div2d(counts_tro, counts_gmi)


im1 = lsm_13 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_13[:, :], tb_gmi[:, :])
div_13 = get_div2d(counts_tro, counts_gmi)

im1 = lsm_14 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_14[:, :], tb_gmi[:, :])
div_14 = get_div2d(counts_tro, counts_gmi)

im1 = stype_aro == imask
counts_aro, counts_gmi = get_hist2d(ta_aro[:, :], tb_gmi[:, :])
div_aro = get_div2d(counts_aro, counts_gmi)

im1 = stype_normal == imask
counts_aro, counts_gmi = get_hist2d(ta_normal[:, :], tb_gmi[:, :])
div_nor = get_div2d(counts_aro, counts_gmi)

print (div_tro, div_11, div_12, div_13, div_14, div_aro, div_nor)








    

#%%
#KL divergence


# water
imask = 0
im = lsm_gmi == 1
im1 = stype_tro == imask
counts_tro, counts_gmi = get_hist2d(ta_tro[im1, :], tb_gmi[im, :])
div_tro = scipy.stats.entropy(counts_tro.ravel(), counts_gmi.ravel())

im1 = lsm_11 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_11[im1, :], tb_gmi[im, :])
div_11 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())

im1 = lsm_12 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_12[im1, :], tb_gmi[im, :])
div_12 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())



im1 = lsm_13 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_13[im1, :], tb_gmi[im, :])
div_13 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = lsm_14 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_14[im1, :], tb_gmi[im, :])
div_14 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = stype_aro == imask
counts_aro, counts_gmi = get_hist2d(ta_aro[im1, :], tb_gmi[im, :])
div_aro = scipy.stats.entropy( counts_gmi.ravel(), counts_aro.ravel())


im1 = stype_normal == imask
counts_aro, counts_gmi = get_hist2d(ta_normal[im1, :], tb_gmi[im, :])
div_nor = scipy.stats.entropy( counts_gmi.ravel(), counts_aro.ravel())



print (div_tro, div_11, div_12, div_13, div_14, div_aro, div_nor)
#%%
# land
imask = 1
im  = (lsm_gmi >= 3) & (lsm_gmi <= 7) 
im1 = stype_tro == imask
counts_tro, counts_gmi = get_hist2d(ta_tro[im1, :], tb_gmi[im, :])
div_tro = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = lsm_11 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_11[im1, :], tb_gmi[im, :])
div_11 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = lsm_12 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_12[im1, :], tb_gmi[im, :])
div_12 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())



im1 = lsm_13 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_13[im1, :], tb_gmi[im, :])
div_13 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = lsm_14 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_14[im1, :], tb_gmi[im, :])
div_14 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = stype_aro == imask
counts_aro, counts_gmi = get_hist2d(ta_aro[im1, :], tb_gmi[im, :])
div_aro = scipy.stats.entropy( counts_gmi.ravel(), counts_aro.ravel())


im1 = stype_normal == imask
counts_aro, counts_gmi = get_hist2d(ta_normal[im1, :], tb_gmi[im, :])
div_nor = scipy.stats.entropy( counts_gmi.ravel(), counts_aro.ravel())



print (div_tro, div_11, div_12, div_13, div_14, div_aro, div_nor)

#%%

# all

counts_tro, counts_gmi = get_hist2d(ta_tro[:, :], tb_gmi[:, :])
div_tro = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = lsm_11 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_11[:, :], tb_gmi[:, :])
div_11 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = lsm_12 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_12[:, :], tb_gmi[:, :])
div_12 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())



im1 = lsm_13 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_13[:, :], tb_gmi[:, :])
div_13 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = lsm_14 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_14[:, :], tb_gmi[:, :])
div_14 = scipy.stats.entropy( counts_gmi.ravel(), counts_tro.ravel())


im1 = stype_aro == imask
counts_aro, counts_gmi = get_hist2d(ta_aro[:, :], tb_gmi[:, :])
div_aro = scipy.stats.entropy( counts_gmi.ravel(), counts_aro.ravel())


im1 = stype_normal == imask
counts_aro, counts_gmi = get_hist2d(ta_normal[:, :], tb_gmi[:, :])
div_nor = scipy.stats.entropy( counts_gmi.ravel(), counts_aro.ravel())



print (div_tro, div_11, div_12, div_13, div_14, div_aro, div_nor)
#%%





fig, ax = plt.subplots(1, 2, figsize = [16, 8])
ax = ax.ravel()
plt.tight_layout() 

imask = 0
sigma = 1.5
# TRO
#im  = (lsm_gmi >= 3) & (lsm_gmi <= 7) 
im = lsm_gmi == 1
im1 = stype_tro == imask
counts_tro, counts_gmi = get_hist2d(ta_tro[im1, :], tb_gmi[im, :])
a11 = scipy.stats.entropy(counts_gmi, counts_tro)
a12 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[1]
a12 = gaussian_filter(a12, sigma)
a11 = gaussian_filter(a11, sigma)
ax[0].plot(xcenter, a12**1, label = r"$\rho$ = 1", c= cc, linewidth = 2.5)
#ax[1].plot(ycenter, a11**1, label =r"$\rho$ = 1", c= cc, linewidth = 2.5)


# ARO == 1.1
im1 = lsm_11 == imask

counts_tro, counts_gmi = get_hist2d(ta_tro_11[im1, :], tb_gmi[im, :])
a11_11 = scipy.stats.entropy(counts_gmi, counts_tro)
a12_11 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[0]
a12_11 = gaussian_filter(a12_11, sigma)
a11_11 = gaussian_filter(a11_11, sigma)
ax[0].plot(xcenter, a12_11**1, label = r"$\rho$ = 1.1", c= cc, linewidth = 2.5)
#ax[1].plot(ycenter, a11_11**1, label = r"$\rho$ = 1.1", c= cc, linewidth = 2.5)

# ARO == 1.2
im1 = lsm_12 == imask

counts_tro, counts_gmi = get_hist2d(ta_tro_12[im1, :], tb_gmi[im, :])
a11_12 = scipy.stats.entropy(counts_gmi, counts_tro)
a12_12 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[2]
a12_12 = gaussian_filter(a12_12, sigma)
a11_12 = gaussian_filter(a11_12, sigma)
ax[0].plot(xcenter, a12_12**1, label = r"$\rho$ = 1.2", c= cc, linewidth = 2.5)
#ax[1].plot(ycenter, a11_12**1, label = r"$\rho$ = 1.2", c= cc, linewidth = 2.5)


# ARO == 1.3

im1 = lsm_13 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_13[im1, :], tb_gmi[im, :])
a11_13 = scipy.stats.entropy(counts_gmi, counts_tro)
a12_13 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[3]
a12_13 = gaussian_filter(a12_13, sigma)
a11_13 = gaussian_filter(a11_13, sigma)
ax[0].plot(xcenter, a12_13**1, label = r"$\rho$ = 1.3", c= cc, linewidth = 2.5)
#ax[1].plot(ycenter, a11_13**1, label = r"$\rho$ = 1.3", c= cc, linewidth = 2.5)


# ARO == 1.4
im1 = lsm_14 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_14[im1, :], tb_gmi[im, :])
a11_14 = scipy.stats.entropy(counts_gmi, counts_tro)
a12_14 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[4]
a12_14 = gaussian_filter(a12_14, sigma)
a11_14 = gaussian_filter(a11_14, sigma)
ax[0].plot(xcenter, a12_14**1, label = r"$\rho$ = 1.4", c= cc, linewidth = 2.5)
#ax[1].plot(ycenter, a11_14**1, label = r"$\rho$ = 1.4", c= cc, linewidth = 2.5)


# ARO
im1 = stype_aro == imask
counts_aro, counts_gmi = get_hist2d(ta_aro[im1, :], tb_gmi[im, :])
#a1 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 0)
#a2 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 1)

a1 = scipy.stats.entropy(counts_gmi, counts_aro)
a2 = scipy.stats.entropy(counts_gmi.T, counts_aro.T)

cc = cols[5]
a2 = gaussian_filter(a2, sigma)
a1 = gaussian_filter(a1, sigma)
ax[0].plot(xcenter, a2**1, label = r"$\rho\in$ U(1, 1.4)",  color =  cc, linewidth = 2.5)
#ax[1].plot(ycenter, a1**1, label = r"$\rho\in$ U(1, 1.4)",  color = cc, linewidth = 2.5)


# ARO
im1 = stype_normal == imask
counts_aro, counts_gmi = get_hist2d(ta_normal[im1, :], tb_gmi[im, :])
#a1 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 0)
#a2 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 1)

a1 = scipy.stats.entropy(counts_gmi, counts_aro)
a2 = scipy.stats.entropy(counts_gmi.T, counts_aro.T)

cc = cols[6]

a2 = gaussian_filter(a2, sigma)
a1 = gaussian_filter(a1, sigma)
ax[0].plot(xcenter, a2**1, label = r"$\rho\in$ $\mathcal{N}$(1.2, 0.12)",  color =  cc, linewidth = 2.5)
#ax[1].plot(ycenter, a1**1, label =  r"$\rho\in$ $\mathcal{N}$(1.2, 0.12)",  color = cc, linewidth = 2.5)



#ax[0].set_xlabel("TB 166V GHz [K]")
#ax[1].set_xlabel("Polarisation difference [K]")

ax[0].set_ylabel("Divergence")
#ax[1].set_ylabel("Divergence")

for i in range(2):
    ax[i].grid("on", alpha = 0.3)
    

    
#fig.savefig("divergence_water.pdf", bbox_inches = "tight")  

# calculate divergences with land -------------------------------

xbins = np.arange(80, 280, 5)
ybins = np.arange(-5, 30, 1.5)


xcenter = (xbins[1:] + xbins[:-1])/2
ycenter = (ybins[1:] + ybins[:-1])/2


imask = 1
# TRO
im  = (lsm_gmi >= 3) & (lsm_gmi <= 7) 
#im = lsm_gmi == 1
im1 = stype_tro == imask
counts_tro, counts_gmi = get_hist2d(ta_tro[im1, :], tb_gmi[im, :])
a11 = scipy.stats.entropy(counts_gmi, counts_tro)
a12 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[1]
a12 = gaussian_filter(a12, sigma)
a11 = gaussian_filter(a11, sigma)
ax[1].plot(xcenter, a12**1, label = r"$\rho$ = 1", c= cc, linewidth = 2.5)
#ax[3].plot(ycenter, a11**1, label =r"$\rho$ = 1", c= cc, linewidth = 2.5)


# ARO == 1.1
im1 = lsm_11 == imask

counts_tro, counts_gmi = get_hist2d(ta_tro_11[im1, :], tb_gmi[im, :])
a11_11 = scipy.stats.entropy(counts_gmi, counts_tro)
a12_11 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[0]
a12_11 = gaussian_filter(a12_11, sigma)
a11_11 = gaussian_filter(a11_11, sigma)
ax[1].plot(xcenter, a12_11**1, label = r"$\rho$ = 1.1", c= cc, linewidth = 2.5)
#ax[3].plot(ycenter, a11_11**1, label = r"$\rho$ = 1.1", c= cc, linewidth = 2.5)


# ARO == 1.2
im1 = lsm_12 == imask

counts_tro, counts_gmi = get_hist2d(ta_tro_12[im1, :], tb_gmi[im, :])
a11_12 = scipy.stats.entropy(counts_gmi, counts_tro)
a12_12 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[2]
a12_12 = gaussian_filter(a12_12, sigma)
a11_12 = gaussian_filter(a11_12, sigma)
ax[1].plot(xcenter, a12_12**1, label = r"$\rho$ = 1.2", c= cc, linewidth = 2.5)
#ax[3].plot(ycenter, a11_12**1, label = r"$\rho$ = 1.2", c= cc, linewidth = 2.5)


# ARO == 1.3

im1 = lsm_13 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_13[im1, :], tb_gmi[im, :])
a11_13 = scipy.stats.entropy(counts_gmi, counts_tro)
a12_13 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[3]
a12_13 = gaussian_filter(a12_13, sigma)
a11_13 = gaussian_filter(a11_13, sigma)
ax[1].plot(xcenter, a12_13**1, label = r"$\rho$ = 1.3", c= cc, linewidth = 2.5)
#ax[3].plot(ycenter, a11_13**1, label = r"$\rho$ = 1.3", c= cc, linewidth = 2.5)


# ARO == 1.4
im1 = lsm_14 == imask
counts_tro, counts_gmi = get_hist2d(ta_tro_14[im1, :], tb_gmi[im, :])
a11_14 = scipy.stats.entropy(counts_gmi, counts_tro)
a12_14 = scipy.stats.entropy( counts_gmi.T, counts_tro.T)

cc = cols[4]
a12_14 = gaussian_filter(a12_14, sigma)
a11_14 = gaussian_filter(a11_14, sigma)
ax[1].plot(xcenter, a12_14**1, label = r"$\rho$ = 1.4", c= cc, linewidth = 2.5)
#ax[3].plot(ycenter, a11_14**1, label = r"$\rho$ = 1.4", c= cc, linewidth = 2.5)


# ARO
im1 = stype_aro == imask
counts_aro, counts_gmi = get_hist2d(ta_aro[im1, :], tb_gmi[im, :])
#a1 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 0)
#a2 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 1)

a1 = scipy.stats.entropy(counts_gmi, counts_aro)
a2 = scipy.stats.entropy(counts_gmi.T, counts_aro.T)

cc = cols[5]
a2 = gaussian_filter(a2, sigma)
a1 = gaussian_filter(a1, sigma)
ax[1].plot(xcenter, a2**1, label = r"$\rho\in$ U(1, 1.4)",  color =  cc, linewidth = 2.5)
#ax[3].plot(ycenter, a1**1, label = r"$\rho\in$ U(1, 1.4)",  color = cc, linewidth = 2.5)


# ARO
im1 = stype_normal == imask
counts_aro, counts_gmi = get_hist2d(ta_normal[im1, :], tb_gmi[im, :])
#a1 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 0)
#a2 = scipy.stats.entropy(counts_aro, counts_gmi, axis = 1)

a1 = scipy.stats.entropy(counts_gmi, counts_aro)
a2 = scipy.stats.entropy(counts_gmi.T, counts_aro.T)

cc = cols[6]

a2 = gaussian_filter(a2, sigma)
a1 = gaussian_filter(a1, sigma)
ax[1].plot(xcenter, a2**1, label = r"$\rho\in$ $\mathcal{N}$(1.2, 0.12)",  color =  cc, linewidth = 2.5)
#ax[3].plot(ycenter, a1**1, label =  r"$\rho\in$ $\mathcal{N}$(1.2, 0.12)",  color = cc, linewidth = 2.5)



ax[0].set_xlabel("TB 166V GHz [K]")
ax[1].set_xlabel("TB 166V GHz [K]")
#ax[3].set_xlabel("Polarisation difference [K]")

ax[0].set_ylabel("Divergence")
#ax[1].set_ylabel("Divergence")

for i in range(2):
    ax[i].grid("on", alpha = 0.3)
    ax[i].set_ylim([0, 8.8])  
    
#ax[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
 
    

  
ax[0].text(78, 8.9, "a)")   
    
ax[1].text(78, 8.9, "b)")  

ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
   
fig.savefig("divergence_land_water.pdf", bbox_inches = "tight")  


#%%

xbins = np.arange(50, 250, 2)
ybins = np.arange(-5, 30, 0.8)

fig, ax = plt.subplots(1, 1, figsize = [8, 8])
counts, xbins, ybins=np.histogram2d(ta_aro[:, 0], ta_aro[:, 0] - ta_aro[:, 1],  
                                        bins=(xbins, ybins), density = True)         

ax.plot(xbins[1:], ybins[np.argmax(counts, axis = 1)], label ="ARO")

counts, xbins, ybins=np.histogram2d(ta_tro[:, 0], ta_tro[:, 0] - ta_tro[:, 1],  
                                        bins=(xbins, ybins), density = True)   


ax.plot(xbins[1:], ybins[np.argmax(counts, axis = 1)], label ="TRO")      

counts, xbins, ybins=np.histogram2d(ta_tro_12[:, 0], ta_tro_12[:, 0] - ta_tro_12[:, 1],  
                                        bins=(xbins, ybins), density = True)   

ax.plot(xbins[1:], ybins[np.argmax(counts, axis = 1)], label ="TRO 12")      


counts, xbins, ybins=np.histogram2d(ta_tro_13[:, 0], ta_tro_13[:, 0] - ta_tro_13[:, 1],  
                                        bins=(xbins, ybins), density = True)   

ax.plot(xbins[1:], ybins[np.argmax(counts, axis = 1)], label ="TRO 13")     


counts, xbins, ybins=np.histogram2d(ta_tro_14[:, 0], ta_tro_14[:, 0] - ta_tro_14[:, 1],  
                                        bins=(xbins, ybins), density = True)   

ax.plot(xbins[1:], ybins[np.argmax(counts, axis = 1)], label ="TRO 14") 


counts, xbins, ybins=np.histogram2d(ta_normal[:, 0], ta_normal[:, 0] - ta_normal[:, 1],  
                                        bins=(xbins, ybins), density = True)   

#ax.plot(xbins[1:], np.max(counts, axis = 1), label ="TRO normal")         



counts, xbins, ybins=np.histogram2d(tb_gmi[im,  0].ravel(), 
           tb_gmi[im, 0].ravel() - tb_gmi[im, 1].ravel(),  
                                        bins=(xbins, ybins), density = True)

ax.plot(xbins[1:], ybins[np.argmax(counts, axis = 1)], label ="GMI")    
  
ax.legend()

#ax.set_yscale("log")



#%%


fig, ax = plt.subplots(1, 2, figsize = [16, 8])
ax = ax.ravel()

imask = 0
im = (lsm_gmi == 1) #| 
#im = (lsm_gmi >= 3) & (lsm_gmi <= 7)
ax[0].scatter(tb_gmi[im, 0], tb_gmi[im, 0] - tb_gmi[im, 1], 
              s = 7, label = "Observations", color = "k")

im = stype_tro == imask
ax[0].scatter(ta_tro[im, 0], ta_tro[im, 0] - ta_tro[im, 1], 
              s = 7, label = r"$\rho$ = 1.0", color = cols[1])

im = lsm_11 == imask
ax[0].scatter(ta_tro_11[im, 0], ta_tro_11[im, 0] - ta_tro_11[im, 1], 
              s = 7, label = r"$\rho$ = 1.1", color = cols[0])


im = lsm_12 == imask
ax[0].scatter(ta_tro_12[im, 0], ta_tro_12[im, 0] - ta_tro_12[im, 1], s = 7, 
              label = r"$\rho$ = 1.2",  color = cols[2])

im = lsm_13  == imask
ax[0].scatter(ta_tro_13[im, 0], ta_tro_13[im, 0] - ta_tro_13[im, 1],
              s = 7, label = r"$\rho$ = 1.3",  color = cols[3])

im = lsm_14 == imask
ax[0].scatter(ta_tro_14[im, 0], ta_tro_14[im, 0] - ta_tro_14[im, 1], s = 7, 
              label = r"$\rho$ = 1.4",  color = cols[4])



imask = 1
#im = (lsm_gmi == 1) #| 
im = (lsm_gmi >= 3) & (lsm_gmi <= 7)
ax[1].scatter(tb_gmi[im, 0], tb_gmi[im, 0] - tb_gmi[im, 1], 
              s = 7, label = "Observations", color = "k")

im = stype_tro == imask
ax[1].scatter(ta_tro[im, 0], ta_tro[im, 0] - ta_tro[im, 1], 
              s = 7, label = "Simulations", color = cols[1])



im = lsm_11 == imask
ax[1].scatter(ta_tro_11[im, 0], ta_tro_11[im, 0] - ta_tro_11[im, 1], 
              s = 7, label = r"$\rho$ = 1.1", color = cols[0])


im = lsm_12 == imask
ax[1].scatter(ta_tro_12[im, 0], ta_tro_12[im, 0] - ta_tro_12[im, 1], s = 7, 
              label = r"$\rho$ = 1.2",  color = cols[2])

im = lsm_13  == imask
ax[1].scatter(ta_tro_13[im, 0], ta_tro_13[im, 0] - ta_tro_13[im, 1],
              s = 7, label = r"$\rho$ = 1.3",  color = cols[3])

im = lsm_14 == imask
ax[1].scatter(ta_tro_14[im, 0], ta_tro_14[im, 0] - ta_tro_14[im, 1], s = 7, 
              label = r"$\rho$ = 1.4",  color = cols[4])

#im = (lsm_gmi == 1) #| ((lsm_gmi >= 3) & (lsm_gmi <= 7))

#ax[3].scatter(tb_gmi[im, 0][::10], tb_gmi[im, 0][::10] - tb_gmi[im, 1][::10], s = 10, label = "GMI")
#im = stype_normal == 0
#ax[3].scatter(ta_normal[im, 0], ta_normal[im, 0] - ta_normal[im, 1], s = 10, label = "ARO")

for i in range(2):
    ax[i].grid("on", alpha = 0.2)
    ax[i].set_xlim([81, 270])
    ax[i].set_ylim([-3, 35])    

ax[0].legend(markerscale=2)


ax[0].set_xlabel("TB 166V GHz [K]") 
ax[1].set_xlabel("TB 166V GHz [K]")

ax[0].set_ylabel("Polarisation difference [K]")   

ax[0].set_title("Water")
ax[1].set_title("Land")
ax[0].text(80, 35.3, "a)")
ax[1].text(80, 35.3, "b)")
fig.savefig("PD_water_varying_rho_water_land.png", bbox_inches = "tight")  


#%% 2d histogram

xbins = np.arange(125, 250, 4)
ybins = np.arange(-5, 25, 0.5)


fig, ax = plt.subplots(1, 3, figsize = [24, 8])
ax = ax.ravel()

colors = iter([plt.cm.tab20(i) for i in range(6)])

c1 = [next(colors)]
c2 = [next(colors)]

im  = tb_gmi[:,  0] > -1
im1 = ta_tro[:, 0] > -1

ax[0].scatter(tb_gmi[im,  0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im,  1].ravel()[::10], 
           label = 'Observation', c = c2, s = 2)




counts, xbins, ybins=np.histogram2d(tb_gmi[im,  0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im, 1].ravel()[::10],  
                                        bins=(xbins, ybins))



sigma = 0.08 # this depends on how noisy your data is, play with it!

ax[0].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'black', locator=ticker.LogLocator(), alpha = 0.5)



counts, xbins, ybins=np.histogram2d(ta_tro[im1, 0], ta_tro[im1, 0] - ta_tro[im1, 1],  
                                        bins=(xbins, ybins))

counts = gaussian_filter(counts, sigma)
ax[0].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'red', locator=ticker.LogLocator(), alpha = 0.5)


#ax[0].scatter(ta_tro[:, 0][::10], ta_tro[:, 0][::10] - ta_tro[:, 1][::10], 
#           label = r"Simulation $\rho$ = 1" , c = c2, s = 2)

cs_sc = ax[1].scatter(tb_gmi[im, 0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im, 1].ravel()[::10], 
           label = 'Observation', c = c2, s = 2)





counts, xbins, ybins=np.histogram2d(tb_gmi[im, 0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im, 1].ravel()[::10],  
                                        bins=(xbins, ybins))

counts = gaussian_filter(counts, sigma)
ax[1].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'black', locator=ticker.LogLocator(), 
                alpha = 0.5, label = "Observation")

im1 = ta_tro_13[:, 0] > -1
counts, xbins, ybins=np.histogram2d(ta_tro_13[im1, 0], ta_tro_13[im1, 0] - ta_tro_13[im1, 1],  
                                        bins=(xbins, ybins))
counts = gaussian_filter(counts, sigma)
ax[1] .contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'red', locator=ticker.LogLocator(), 
                alpha = 0.5, label =   r"Simulation [\rho = 1]")



ax[2].scatter(tb_gmi[im, 0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im,  1].ravel()[::10], 
           label = 'Observation', c = c2, s = 2)


counts, xbins, ybins=np.histogram2d(tb_gmi[im, 0].ravel()[::10], 
           tb_gmi[im, 0].ravel()[::10] - tb_gmi[im,  1].ravel()[::10],  
                                        bins=(xbins, ybins))

counts = gaussian_filter(counts, sigma)
cs_gmi = ax[2].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'black', locator=ticker.LogLocator(),
                alpha = 0.5, label = r"Simulation [\rho = 1.2]")


im1 = ta_aro[:, 0] > -1
counts, xbins, ybins=np.histogram2d(ta_aro[im1, 0], ta_aro[im1, 0] - ta_aro[im1, 1],  
                                        bins=(xbins, ybins))

counts = gaussian_filter(counts, sigma)
cs = ax[2].contour(counts.transpose(),extent=[xbins.min(),xbins.max(),
                ybins.min(),ybins.max()],linewidths=3,
                linestyles='solid', colors = 'red', locator=ticker.LogLocator(),
                alpha = 0.5, label = r"Simulation [\rho \in U[1, 1.4]]")


lines = [ cs.collections[0], cs_gmi.collections[0]]


#ax[1].scatter(ta_tro_13[:, 0][::5], ta_tro_13[:, 0][::5] - ta_tro_13[:, 1][::5], 
#           label = r"Simulation $\rho$ = 1.3" , c = c2, s = 2)

for i in range(3):
    ax[i].grid("on", alpha = 0.3)
    ax[i].set_xlabel("TB 166V GHz [K]")
    ax[0].set_ylabel("Polarisation difference [K]")
    lgnd = ax[i].legend()
    ax[i].legend(lines, [ "Simulation", "Observation"], loc = 'upper left')
#change the marker size manually for both lines
    lgnd.legendHandles[0]._sizes = [30]
#    lgnd.legendHandles[1]._sizes = [30]

ax[0].set_title(r"$\rho$ = 1")
ax[1].set_title(r"$\rho$ = 1.2")
ax[2].set_title(r"$\rho$ $\in$ U(1, 1.4)")
fig.savefig("PD_166.png", bbox_inches = "tight")
#cc = next(colors)
#ax.scatter(ta_tro_12[:, 0], ta_tro_12[:, 0] - ta_tro_12[:, 1], label = r"$\rho$ = 1.2", c = cc )
    
    
    


