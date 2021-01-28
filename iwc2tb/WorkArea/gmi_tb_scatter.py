#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:02:32 2021

@author: inderpreet
"""

import os
import numpy as np
import glob
from iwc2tb.GMI.GMI import GMI


inpath =  os.path.expanduser('~/Dendrite/Projects/IWP/GMI/GMI_m65_p65')  
 
#matfile = glob.glob("/home/inderpreet/data/temp/OutIcube/*.mat")
matfiles = glob.glob(os.path.join(inpath, "*.mat"))

dardarpath = os.path.expanduser("~/Dendrite/Projects/IWP/GMI/DARDAR_ERA_m65_p65")

# GMI frequencies
freq = ["166.5V", "166.5H", "183+-3", "183+-7"]




