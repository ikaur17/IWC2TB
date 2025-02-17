import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset

class gmiData(Dataset):
    """
    Pytorch dataset for the GMI training data for IWP retrievals

    """
    def __init__(self, path, inChannels,
                 batch_size = None,
                 ocean = False,
                 test_data = False):
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
        channels = self.file.variables["channels"][:]
        self.stype = ta.stype
        self.lon   = ta.lon
        self.lat   = ta.lat
        self.iwp   = ta.iwp
        self.rwp   = ta.rwp
        self.t0    = ta.t0
        
        #self.p0    = ta.p0
        #self.z0    = ta.z0 

        self.channels = inChannels       
        idx = []
        
        for i in range(len(inChannels)):

            idx.append(np.argwhere(channels == inChannels[i])[0][0]) 
                                                                            
        self.ocean = ocean
        self.index = idx
        
        C = []
        
        C.append(TB)

        C.append(self.t0.reshape(-1, 1))
        C.append(self.lat.reshape(-1, 1))
        C.append(self.lon.reshape(-1, 1))  
        C.append(self.stype.reshape(-1, 1))

        x = np.float32(np.concatenate(C, axis = 1))

#         im = []
#         for i in self.index:
#             im.append(TB.mask[i, :])
#         im = np.stack(im, axis = 1)
# #        print (im.shape, np.sum(im))
#         im = np.logical_or.reduce(im, axis = 1)	
        
	    #store mean and std to normalise data
        
        # if self.ocean:
        #    il = self.stype == 0
        #    im = np.logical_and(~im, il) 
        # else:
        #     im = ~im
            
        x_noise = self.add_noise(x[:, :4], self.index)
        
        x[:, :4] = x_noise
  
        self.std = np.std(x, axis = 0)[:5]
        self.mean = np.mean(x, axis = 0)[:5]  
        
        self.y = np.float32(self.iwp)
   
        #self.y_noise = self.add_noise(self.y, self.itarget) 

        self.x = x.data
        self.y = self.y
        
        #self.y_noise = self.y_noise.data[im]
        # self.lsm = np.float32(self.lsm[im])
        # self.lsm = np.expand_dims(self.lsm, axis = 1)
        # self.im  = im
        
        # if test_data:
        #     test_file_path = path.replace("test", "test_noisy")
        #     test_file = netCDF4.Dataset(test_file_path, mode = "r")
        #     print (test_file_path)
        #     print (path)
        #     ta = test_file.variables["ta"]
        #     TB_noise = ta[:]
                                                                     
        #     C = []
            
        #     for ic in self.index:
        #         C.append(TB_noise[1, ic, :])
    
        #     self.x = np.float32(np.stack(C, axis = 1))
        #     self.std = np.std(self.x[im, :], axis = 0)
        #     self.mean = np.mean(self.x[im, :], axis = 0)   
       
        #     self.y_noise =  np.float32(TB_noise[0, self.itarget[0], :])

        #     self.x = self.x.data[im, :]
        #     self.y_noise = self.y_noise.data[im]
            


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
            self.x  = self.x[indices, :]
            self.y  = self.y[indices]

        if self.batch_size is None:
            return (torch.tensor(self.x[[i], :]),
                    torch.tensor(self.y[[i]]))
        else:
            i_start = self.batch_size * i
            i_end   = self.batch_size * (i + 1)    
            
            x       = self.x[i_start : i_end, :].copy()
            x_noise = np.float32(self.add_noise(x[:, :4], self.index))
            
            x[:, :4]      = x_noise
            x_norm        = x.copy()
            x_norm[:, :5]= np.float32(self.normalise(x[:, :5]))

            # if not self.ocean:
            #     x_norm = np.concatenate((x_norm, self.lsm[i_start : i_end, :]), axis = 1)
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
            for ic in range(len(self.index)):
                noise = np.random.normal(0, nedt_subset[ic], size_TB)
                x_noise[:, ic] += noise
        else:
                noise = np.random.normal(0, nedt_subset, size_TB)
                x_noise[:] += noise
        return x_noise    
 
    def normalise(self, x):
        """
        normalise the input data with mean and standard deviation
        Args:
            x
        Returns :
            x_norm
        """          

        x_norm = (x - self.mean)/self.std   
            
        return x_norm 
        
