#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:51:03 2021

@author: inderpreet
"""


def lsm_gmi2arts(lsm):
        
    stype = lsm.copy()
    stype[lsm == 1] = 0
    stype[lsm == 2] = 3
    stype[lsm == 3] = 1
    stype[lsm == 4] = 1
    stype[lsm == 5] = 1
    stype[lsm == 6] = 1
    stype[lsm == 7] = 1
    stype[lsm == 8] = 2
    stype[lsm == 9] = 2
    stype[lsm == 10] = 2
    stype[lsm == 11] = 2
    stype[lsm == 12] = 0
    stype[lsm == 13] = 4
    stype[lsm == 14] = 6
    return stype    