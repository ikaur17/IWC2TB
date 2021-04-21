import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset

class gmiData(Dataset):
    """
    Pytorch dataset for the GMI training data for IWP retrievals

    """
    def __init__(self, path, 
                 inputs,
                 outputs,
                 batch_size = None,
                 latlims = None,
                 std = None,
                 mean = None,                 
                 log = False):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size

        """
        super().__init__()
        
        self.batch_size = batch_size

        self.file = netCDF4.Dataset(path, mode = "r")

        ta = self.file.variables["ta"]
        TB = ta[:]

        self.stype = ta.stype
        self.lon   = ta.lon
        self.lat   = ta.lat
        self.iwp   = ta.iwp
        self.rwp   = ta.rwp
        self.t0    = ta.t0
        self.z0    = ta.z0
        self.wvp   = ta.wvp
        #self.lst   = ta.lst
        self.t2m   = ta.t2m
        self.p0    = ta.p0

        
        all_inputs = [TB, self.t0.reshape(-1, 1), 
                      self.lon.reshape(-1, 1), self.lat.reshape(-1, 1),
                      self.stype.reshape(-1, 1), self.t2m.reshape(-1, 1), 
                      self.wvp.reshape(-1, 1), self.z0.reshape(-1, 1),
                      self.p0.reshape(-1, 1)] 
        
        inputnames = np.array(["ta", "t0",
                               "lon", "lat", "stype",
                               "t2m", "wvp", "z0", 
                               "p0"])
        

        outputnames = np.array(["iwp", "rwp", "wvp"])
        
        idy         = np.argwhere(outputnames == outputs)[0][0]
        
        self.inputs = inputs       
        idx = []
        
        for i in range(len(inputs)):
            idx.append(np.argwhere(inputnames == inputs[i])[0][0]) 
                                                                            

        self.index = idx
        self.chindex = [0, 1, 2, 3]
        C = []
        for i in idx:
            C.append(all_inputs[i])
            
        x = np.float32(np.concatenate(C, axis = 1))
        
        if latlims is not None:            
            ilat = (np.abs(self.lat) >= latlims[0]) & (np.abs(self.lat) <= latlims[1])
            x         = x[ilat, :]
            self.iwp  = self.iwp[ilat]
            self.lon  = self.lon[ilat] 
            self.wvp  = self.wvp[ilat]
            self.rwp  = self.rwp[ilat]
            #self.lst  = self.lst[ilat]
            self.t2m  = self.t2m[ilat]
            self.t0   = self.t0[ilat]
            self.z0   = self.z0[ilat]
            self.p0   = self.p0[ilat]            
            
        all_outputs = [self.iwp, self.rwp, self.wvp]    
            
        if std is not None:
            self.std = std
        else:
            self.std = np.std(x, axis = 0)
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(x, axis = 0)
        
        self.y = np.float32(all_outputs[idy])
        
        self.x = x.data
        
        self.y[self.y == 0] = 1e-12

        if log == True:
            self.y = np.log(self.y)            

        self.file.close()

    def __len__(self):
        """
        The number of entries in the training data. This is part of the
        pytorch interface for datasets.

        Return:
            int: The number of samples in the data set
        """
        if self.batch_size is None:
            return self.x.shape[0]
        else:
            return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i: The index of the sample to return
        """
        if (i == 0):

            indices = np.random.permutation(self.x.shape[0])
            self.x   = self.x[indices, :]
            self.y   = self.y[indices]
            self.lon = self.lon[indices]
            self.lat = self.lat[indices]
            #self.lst = self.lst[indices]

        if self.batch_size is None:
            return (torch.tensor(self.x[[i], :]),
                    torch.tensor(self.y[[i]]))
        else:
            i_start = self.batch_size * i
            i_end   = self.batch_size * (i + 1)    
            
            x       = self.x[i_start : i_end, :].copy()
            x_noise = np.float32(self.add_noise(x[:, :4], self.chindex))
            
            x[:, :4]      = x_noise
            x_norm        = x.copy()
            x_norm        = np.float32(self.normalise_std(x))
            
            return (torch.tensor(x_norm),
                    torch.tensor(self.y[i_start : i_end]))
        
        
    def add_noise(self, x, index):        
        """
        Gaussian noise is added to every measurement before used 
        for training again.
        
        Args: 
            the input TB in one batch of size (batch_size x number of channels)
        Returns:
            input TB with noise
            
        """
        
        nedt  = np.array([0.70, # 166 V
                          0.65, # 166 H
                          0.56, # 183+-3
                          0.47  # 183+-7                     
                          ])

        
        nedt_subset = nedt[index]
        size_TB = int(x.size/len(nedt_subset))
        x_noise = x.copy()
        if len(index) > 1:
            for ic in range(len(index)):
                noise = np.random.normal(0, nedt_subset[ic], size_TB)
                x_noise[:, ic] += noise
        else:
                noise = np.random.normal(0, nedt_subset, size_TB)
                x_noise[:] += noise
        return x_noise    
 
    def normalise_std(self, x):
        """
        normalise the input data with mean and standard deviation
        Args:
            x
        Returns :
            x_norm
        """          

        x_norm = (x - self.mean)/self.std   
            
        return x_norm 
        
    def normalise_minmax(self, x):

        x_norm = (x - x.min())/(x.max() - x.min())
        return x_norm
