import sys
import numpy as np
import torch 
from torch.utils.data import Dataset

#-------------------------------------------------------
class RBC2dDataset(Dataset):
    ''' Dataset for fields ux, uy, T of 2D RBC.'''
    
    
    #-------------------------------------------------------
    def __init__(self, filepath, tstart, tend, transform = None, fields_use=None):
        
        self.filepath = filepath
        self.transform = transform
        self.tstart   = tstart
        self.tend     = tend
        self.nsamples = tend - tstart
            
        if fields_use is None:
            self.fields_use = [0,1,2]
        else:
            assert len(fields_use) <= 3, f"nfields ({len(fields_use)}) can not be larger than 3 (ux,uy,T)!"
            self.fields_use = fields_use
        self.nfields = len(self.fields_use)
        
        # Import data
        #--------------
        self.data = torch.from_numpy(np.load(filepath))
        self.data = self.data[self.tstart:self.tend,self.fields_use]
        
    #-------------------------------------------------------
    def __getitem__(self, it):            
        ''' Import data and return fields at time step it+1. Note: filenames start at velx.dat_00001, ...'''

        # Get snapshot
        #-------------
        x = self.data[it].clone()

        # Transform data
        #---------------
        if self.transform:
            x = self.transform(x)
            
        return x
    
    #-------------------------------------------------------
    def __len__(self):
        return self.nsamples
    
    #-------------------------------------------------------