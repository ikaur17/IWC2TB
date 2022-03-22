#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:15:34 2021

@author: inderpreet
"""

import numpy as np
import os
import glob
from iwc2tb.GMI.GMI import GMI
from iwc2tb.GMI.GMI_SatData import GMI_Sat
from remove_oversampling_gmi import remove_oversampling_gmi
import random
import scipy
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib as mpl
from iwc2tb.GMI.three_sigma_rule import three_sigma
from scipy.ndimage.filters import gaussian_filter
plt.rcParams.update({'font.size': 18})

#%%
def plot_contours(tb_gmi, ax):
    xbins = np.arange(50, 280, 5)
    ybins = np.arange(-5, 40, 0.5)
    counts, xbins, ybins=np.histogram2d(tb_gmi[:,  0].ravel(),
               tb_gmi[:, 0].ravel() - tb_gmi[:, 1].ravel(),
                                            bins=(xbins, ybins), density = True)
    sigma = 0.1
    counts = gaussian_filter(counts, sigma)
    x, y = np.meshgrid(xbins, ybins)

    # Calculate Percentiles
    CT = counts.T
    CT_perc = np.zeros([2])
    CT_perc[0] = np.percentile(CT, 5)
    CT_perc[1] = np.percentile(CT, 95)

    nb_features = 2
    contourlist = [np.array((0.,0.,0.)) for i in np.arange(nb_features)]

    contourlist[0] = "tab:blue"     # 5%     (red)
    contourlist[1] = np.array((40.,40.,40.))/255       # 95%     (black)
    cmap_contour = mpl.colors.ListedColormap(contourlist)

    colorbins = [CT_perc[0], CT_perc[1]]
    norm_contour = mpl.colors.BoundaryNorm(colorbins, cmap_contour.N)

    ax.contour(x[:-1,:-1], y[:-1,:-1], CT, cmap=cmap_contour,
               norm=norm_contour, linewidths=1.5)


#%%

inpath_gmi   = os.path.expanduser('~/Dendrite/SatData/GMI/L1B/2017/01/')
# GMI simulations
gmifiles = glob.glob(os.path.join(inpath_gmi, "*/*.HDF5"))

random.shuffle(gmifiles)
gmi_sat = GMI_Sat(gmifiles[:22])


key = "lpa"
# GMI frquencies
freq     = ["166.5V", "166.5H",  "183+-7", "183+-3",]

cols = ["#999999", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "lawngreen"]


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
inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/esa')
matfiles1 = glob.glob(os.path.join(inpath_mat, "2009_00*.mat"))
matfiles2 = glob.glob(os.path.join(inpath_mat, "2009_01*.mat"))
matfiles3 = glob.glob(os.path.join(inpath_mat, "2009_02*.mat"))
matfiles4 = glob.glob(os.path.join(inpath_mat, "2009_03*.mat"))
matfiles  = matfiles1 + matfiles2 + matfiles3 + matfiles4

gmi                     = GMI(matfiles[:])
ta_aro                  = gmi.ta_noise
stype_aro               = gmi.stype

th = three_sigma(ta_aro[:, 3])
mask_a = ta_aro[:, 3] < th

ta_aro = ta_aro[mask_a, :]
stype_aro = stype_aro[mask_a]

#%%

inpath_mat   =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65_v1.1/esa_pr_1')
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


#%% making subsets with polarisation ratio


mask = (gmi.pratio[mask_a] > 1.07) & (gmi.pratio[mask_a] < 1.12)
ta_tro_11 = ta_aro[mask]
lsm_11= stype_aro[mask]

mask = (gmi.pratio[mask_a] > 1.27) & (gmi.pratio[mask_a] < 1.33)
ta_tro_13 = ta_aro[mask]
lsm_13 = stype_aro[mask]


mask = (gmi.pratio[mask_a] > 1.17) & (gmi.pratio[mask_a] < 1.23)
ta_tro_12 = ta_aro[mask]
lsm_12 = stype_aro[mask]

mask = (gmi.pratio[mask_a] > 1.35) &  (gmi.pratio[mask_a] <= 1.4)
ta_tro_14 = ta_aro[mask]
lsm_14 = stype_aro[mask]

mask = (gmi.pratio[mask_a] == 1.5)
ta_tro_15 = ta_aro[mask]
lsm_15 = stype_aro[mask]

#%% scatter plot PD vs TBV wrt water/land

cols = plt.cm.Set3

fig, ax = plt.subplots(1, 1, figsize = [8, 8])

imask = 1
im = (lsm_gmi == 1) | ((lsm_gmi >= 3) & (lsm_gmi <= 7))
#im = (lsm_gmi >= 3) & (lsm_gmi <= 7)
ax.scatter(tb_gmi[im, 0], tb_gmi[im, 0] - tb_gmi[im, 1],
              s = 7, label = "Observations", color = "dimgray")

im = stype_tro <= imask
ax.scatter(ta_tro[im, 0], ta_tro[im, 0] - ta_tro[im, 1],
              s = 7, label = r"$\rho$ = 1.0", color = cols(0), )

im = lsm_11 <= imask
ax.scatter(ta_tro_11[im, 0], ta_tro_11[im, 0] - ta_tro_11[im, 1],
              s = 7, label = r"$\rho$ = 1.1", color = cols(1), )


im = lsm_12 <= imask
ax.scatter(ta_tro_12[im, 0], ta_tro_12[im, 0] - ta_tro_12[im, 1], s = 7,
              label = r"$\rho$ = 1.2",  color = cols(2),)

im = lsm_13 <= imask
ax.scatter(ta_tro_13[im, 0], ta_tro_13[im, 0] - ta_tro_13[im, 1],
              s = 7, label = r"$\rho$ = 1.3",  color = cols(3), )

im = lsm_14 <= imask
ax.scatter(ta_tro_14[im, 0], ta_tro_14[im, 0] - ta_tro_14[im, 1], s = 7,
              label = r"$\rho$ = 1.4",  color = cols(6), )

im = lsm_15 <= imask
ax.scatter(ta_tro_15[im, 0], ta_tro_15[im, 0] - ta_tro_15[im, 1], s = 7,
              label = r"$\rho$ = 1.5",  color = cols(7), )

im = (lsm_gmi == 1) | ((lsm_gmi >= 3) & (lsm_gmi <= 7))
plot_contours(tb_gmi[im, :], ax)

ax.grid("on", alpha = 0.2)
ax.set_xlim([60, 290])
ax.set_ylim([-5, 35])

ax.set_xlabel("TB 166V GHz [K]")
ax.legend(markerscale=3.5, fontsize = 19)
ax.set_ylabel("Polarisation difference [K]")

ax.set_title("Water + Land")
fig.savefig("PD_water_varying_rho_water_land_esa.png", bbox_inches = "tight")
