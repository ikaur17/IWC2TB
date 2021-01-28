#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:26:24 2021

@author: inderpreet
"""

import h5py

class GMI_Sat():
    
    def __init__(self, filename):
        
        self.file = h5py.File(filename, 'r')
        
        
    def get_keys(self) :
        """
        get SDS keys

        Returns
        -------
        list containing names of the SDS names

        """
        #Names of the groups in HDF5 file.
        
        return self.file.keys()

    def get_data(self, parameter):
        """
        get data for the parameter 
        
        Currently only accessing data from high freq channels under "S2"

        Parameters
        ----------
        parameter : string containing name of the SDS variable

        Returns
        -------
        np.array containing data

        """

        return  self.file["S1"]["parameter"]

    @property
    def lat(self):    
        """
        Latitude values

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return self.get_data('Latitude')
    
    @property
    def lon(self): 
        """
        Longitude values

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return self.get_data('Longitude')
    
    @property
    def tb(self): 
        """
        brightness temperature values

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        return self.get_data('Tb')