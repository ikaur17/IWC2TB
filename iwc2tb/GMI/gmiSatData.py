import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset
from iwc2tb.GMI.lsm_gmi2arts import lsm_gmi2arts
from iwc2tb.GMI.swap_gmi_183 import swap_gmi_183

class gmiSatData(Dataset):
    """
    Pytorch dataset for the GMI training data for IWP retrievals

    """
    def __init__(self, gmi, 
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
        
        self.gmi    = gmi

        TB = self.gmi.tb
        
        TB = swap_gmi_183(TB)
        
        self.lsm   = self.gmi.get_lsm()
        self.lon   = self.gmi.lon
        self.lat   = self.gmi.lat
        self.iwp   = self.gmi.iwp
        self.rwp   = self.gmi.rwp
        self.t2m   = self.gmi.t0
        self.wvp   = self.gmi.wvp
        self.lst   = self.gmi.lst
        self.stype = lsm_gmi2arts(self.lsm) 

        
        all_inputs = [TB, self.t2m[:, :, np.newaxis], 
                      self.lon[:, :, np.newaxis], self.lat[:, :, np.newaxis],
                      self.stype[:, :, np.newaxis], self.wvp[:, :, np.newaxis]] 
        
        inputnames = np.array(["ta", "t2m",
                               "lon", "lat", "stype",
                                "wvp"])
        

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
            
        x = np.float32(np.concatenate(C, axis = 2))
        
        ilat = np.logical_and(np.abs(self.lat) >= latlims[0],
                                      np.abs(self.lat) <= latlims[1])
        

        x         = x[ilat[:, 0], :, :]
        self.iwp  = self.iwp[ilat[:, 0], :]
        self.lat  = self.lat[ilat[:, 0], :]
        self.lon  = self.lon[ilat[:, 0], :] 
        self.wvp  = self.wvp[ilat[:, 0], :]
        self.rwp  = self.rwp[ilat[:, 0], :]
        self.lst  = self.lst[ilat[:, 0], :]
        self.t2m  = self.t2m[ilat[:, 0], :]
        self.stype = self.stype[ilat[:, 0], :]
        


            
        all_outputs = [self.iwp, self.rwp, self.wvp]    
            
        if std is not None:
            self.std = std
        else:
            self.std = np.std(x, axis = (0, 1))
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(x, axis = (0, 1))
        
        self.y = np.float32(all_outputs[idy])
        
        self.x = x
        
#        self.y = np.where(self.y ==0, 1e-12)

        if log == True:
            self.y = np.log(self.y)            

#        self.file.close()

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
        # if (i == 0):

        #     indices = np.random.permutation(self.x.shape[0])
        #     self.x   = self.x[indices, :]
        #     self.y   = self.y[indices]
        #     self.lon = self.lon[indices]
        #     self.lat = self.lat[indices]
        #     self.lst = self.lst[indices]

        if self.batch_size is None:
            return (torch.tensor(self.x[i, :, :]),
                    torch.tensor(self.y[i, :, :]))
        else:
            
            i_start = self.batch_size * i
            i_end   = self.batch_size * (i + 1)             
            
            x       = self.x[i_start : i_end, :, :]

#            x_norm        = x.copy()
            x_norm        = np.float32(self.normalise_std(x))
            
            return (torch.tensor(x_norm),
                    torch.tensor(self.y[i_start : i_end, :]))
        
  
 
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
    
                
            
            
        
            
