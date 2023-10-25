import torch
import numpy as np
import h5py
import os
import sys


class AE:
    
    def __init__(self,config,):
        """
        INPUT:
            config   - yaml config file
        """
        
        self.init_from_config(config)
        self.set_up_device()
        self.set_up_save_file()
        
        self.nfields=len(self.fields_use)
        self.total_points = int(self.nx*self.ny*self.nz)
        
        if self.nz > 1:
            self.set_up_3d()
        else:
            self.set_up_2d()
        
    #----------------------------------------------------------
    def init_from_config(self,config):
        
            # Paths
            #-------
            self.datapath     = config['datapath']
            self.base_dir     = config['base_dir']
            try:
                self.subdir_str   = config['subdir_str']
            except KeyError:
                self.subdir_str = config['comment_str']
                
            self.model_path   = config['model_path']
            self.model_name   = config['model_name']
            self.dataset_path = config['dataset_path']
            self.dataset_name = config['dataset_name']
                
            self.filepath_minmax = config['filepath_minmax']
            
            # Data Layout
            #--------------
            #self.nt = config['nt']
            self.nx = config['nx']
            self.ny = config['ny']
            self.nz = config.get('nz', 1)
            self.aspect = config.get('aspect',4)
            self.fields_use = tuple(config['fields_use'])
            
            # AE Layout
            #--------------
            self.num_epochs = config['num_epochs']
            self.encoding_dim  = config['encoding_dim']
            self.learning_rate = config['learning_rate']
            self.lr_lambda     = config['lr_lambda']
            self.weight_decay  = config['weight_decay']
            self.n_epoch_stop = config['n_epoch_stop']
            
            self.latent_activation = config['latent_activation']
            self.activation   = config['activation']
            self.p_conv       = config.get('p_conv',0.15)
            self.p_lin        = config.get('p_lin',0.15)
            self.padding_mode = config.get('padding_mode','zeros')
            self.size_factor  =  config['size_factor']
            self.randomSeed = config['randomSeed']
            
            # Dataloader Layout
            #-------------------
            self.trainingLength   = config.get('trainingLength',8000)
            self.validationLength = config.get('validationLength',1000)
            self.testingLength    = config.get('testingLength',1000)
            self.batch_size       = config['batch_size']
            self.shuffle_train    = config['shuffle_train']
            self.shuffle_val      = config['shuffle_val']
            
            # Misc. 
            #--------------
            self.use_cuda = config['use_cuda']
            
    #----------------------------------------------------------
    def set_up_3d(self):
        
            self.lta_dim = (0,2,3)
            
            xTab = torch.linspace(-self.aspect/2,self.aspect/2,self.nx)
            yTab = torch.linspace(-self.aspect/2,self.aspect/2,self.ny)
            self.vertical_coord = torch.linspace(0,1,self.nz)
            Z,_,_  = torch.meshgrid(self.vertical_coord,yTab,xTab,indexing='ij')
            self.diffusive_profile = 1 - Z.reshape(self.nz,self.ny,self.nx)
    
    #----------------------------------------------------------
    def set_up_2d(self):
            self.lta_dim = (0,2)
            
            xTab = torch.linspace(-self.aspect/2,self.aspect/2,self.nx)
            self.vertical_coord = torch.linspace(0,1,self.ny)
            Y,_  = torch.meshgrid(self.vertical_coord,xTab,indexing='ij')
            self.diffusive_profile = 1-Y.reshape(self.nz,self.ny,self.nx)
          
    #----------------------------------------------------------
    def set_up_device(self):
        """
        Sets up cpu or gpu device for AE training.
        """
        
        if self.use_cuda:
            
            if sys.platform != 'darwin':
                # Linux / Windows
                assert torch.cuda.is_available, "CUDA not found"
                self.device = 'cuda'
                self.cuda = torch.device('cuda:0')
            else:
                # Mac OS
                self.device = 'mps'
                self.cuda = torch.device('mps')
        else:
            self.device = 'cpu'
            self.cuda = 'cpu'
            
    #----------------------------------------------------------
    def set_up_save_file(self):
        '''
        Sets up checkpoint and history filepaths.
        '''

        batch_size    = self.batch_size
        num_epochs    = self.num_epochs
        encoding_dim  = self.encoding_dim
        learning_rate = self.learning_rate
        lr_lambda     = self.lr_lambda
        weight_decay  = self.weight_decay
        
        lr_str = "{0:.0e}".format(self.learning_rate).replace("-0", "")
        wd_str = "{0:.0e}".format(self.weight_decay).replace("-0", "")
        lr_lambda_str = "{0:.0e}".format(self.lr_lambda).replace("+0", "")

        self.base_name = f'bs{self.batch_size}_epochs{self.num_epochs}_ld{self.encoding_dim}_lr{lr_str}_wd{wd_str}_lambda{lr_lambda_str}'
        
        self.results_dir = os.path.join(self.base_dir, self.model_name, self.dataset_name,self.subdir_str)
        os.makedirs(self.results_dir, exist_ok=True)

        cp_name           = f'best_{self.base_name}.pth'
        history_base_name = f'{self.base_name}.hdf5'
        self.cp_path      = os.path.join(self.results_dir, cp_name)
        self.history_path = os.path.join(self.results_dir, history_base_name)
        
    #----------------------------------------------------------
    def get_minmax_values(self):
        '''
        
        RETURN
            x_min - min. values of all fields
            x_max - max. values of all fields
        '''
        with h5py.File(self.filepath_minmax, 'r') as f:
    
            if self.nz > 1:
                x_min = torch.tensor(f['min'][list(self.fields_use)]).reshape(self.nfields, 1, 1, 1)
                x_max = torch.tensor(f['max'][list(self.fields_use)]).reshape(self.nfields, 1, 1, 1)
            else:
                x_min = torch.tensor(f['min'][list(self.fields_use)]).reshape(self.nfields, 1, 1)
                x_max = torch.tensor(f['max'][list(self.fields_use)]).reshape(self.nfields, 1, 1)
                
        return x_min,x_max
                                 
    #----------------------------------------------------------    
    def save_history(self,history_dict):
        """
        Saves losses, stored in history_dict, to .hdf5 file (self.history_path).
        
        INPUT:
            history_dict - dict which will be stored as hdf5 file
        """
        with h5py.File(self.history_path, 'a') as f:
            for key, values in history_dict.items():
                if key in f:
                    del f[key]
                f.create_dataset(name=key, data=np.array(values), compression='gzip', compression_opts=9)
                
    #----------------------------------------------------------                   
    def save_model_checkpoint(self,model):
        """
        Saves AE model to self.cp_path.
        
        INPUT:
            model - trained PyTorch model that will be saved
        """
        print('Saving as ' + self.cp_path)
        torch.save(model.cpu(), self.cp_path)
    
    #----------------------------------------------------------
    def read_history(self):
        """
        Reads losses, stored in history_dict, to .hdf5 file (self.history_path).
        
        RETURN:
            loss_dict - dict w. AE losses over epochs
        """
        
        loss_dict = dict()
        with h5py.File(self.history_path, 'r') as f:
            for key in f.keys():
                loss_dict[key] = torch.from_numpy(np.array(f[key]))
                
        return loss_dict
                                 
    #----------------------------------------------------------
    #def read_model(self,device='cpu'):
    #    """
    #    Reads saved AE from checkpoint .pth-file.
    #    
    #    RETURN:
    #        model - trained Autoencoder
    #    """
    
     #   model = torch.load(self.cp_path)
     #   if hasattr(model,'module'):
     #       model = model.module
            
     #   return model.to(device)
    
    #----------------------------------------------------------
    def unddo_minmax_transform(self,x,x_min=None,x_max=None):
        """
        Reverses transform_minmax.
        INPUT:
            x - to [0,1] normalized physical fields of shape (batch_size, nfields, nz, ny, nx)
        RETURN:
            x - physical fields with proper scale
        """
        if x_min is None and x_max is None:
            x_min, x_max = self.get_minmax_values()
            
        if self.nz > 1:
            target_shape = (1,self.nfields,1,1,1)
        else:
            target_shape = (1,self.nfields,1,1)    
        
        return x*(x_max.reshape(target_shape)-x_min.reshape(target_shape)) + x_min.reshape(target_shape)
    
    #----------------------------------------------------------  
    def run_model(self,model,data):
        """
        Runs AE model w. provided data.
        
        INPUT:
            model - AE model
            data  - data, shape (batch_size,nfields,nz,ny,nx)
        RETURN:
            encoded - output of model.encoder 
            decoded - output of model.decoder
        """
        #if model was trained on PyTorch < v1.13.1
        for module in model.decoder:
            if isinstance(module,torch.nn.modules.upsampling.Upsample):
                module.recompute_scale_factor = None

        with torch.no_grad():
            model.eval()
            decoded, encoded = model(data)

        encoded = encoded.detach()
        decoded = decoded.detach()

        return encoded, decoded

    #----------------------------------------------------------
    def compute_lta(self,x1,x2=None):
        """
        Returns lateral-time-average profile (<x1^2>A,t)^1/2 or <x1*x2>A,t for given fields x1,x2.
        INPUT:
            x1 - field 1
            x2 - field 2 (optional)
        RETURN:
            lta - lateral-time average <>_A,t
        """

        if self.nz > 1:
            assert x1.shape[1:] == (self.nz,self.ny,self.nx), "Expected x1 to be of shape (nt, nz, ny, nx)"
        
            if x2 is not None:
                assert x2.shape[1:] == (self.nz,self.ny,self.nx), "Expected x2 to be of shape (nt, nz, ny, nx)"
        else:
            assert x1.shape[1:] == (self.ny,self.nx), "Expected x1 to be of shape (nt, ny, nx)"
            
            if x2 is not None:
                assert x2.shape[1:] == (self.ny,self.nx), "Expected x2 to be of shape (nt, ny, nx)"
                
        # Compute lateral time avgerage
        #-------------------------------
        if x2 is None:
            lta = torch.sqrt(torch.mean(x1**2, dim=self.lta_dim)) # Root-mean-square (<x1^2>A,t)^1/2
        else:
            lta = torch.mean(x1*x2, dim=self.lta_dim)            # Covar. <x1*x2>_A,t

        return lta
                                 
    #----------------------------------------------------------
    def compute_nare(self,lta_truth,lta_test):
        """
        Compute normalized average relative error (NARE) of true and inferred lateral-time average profile.
        Definition see e.g.: Heyder & Schumacher PRE 103, 053107 (2021) eqs. (29-30)
        INPUT:
            lta_truth - lateral-time average of ground truth
            lta_test  - lateral-time average of inferred field
        RETURN:
            nare - normalized average relative error of lta profile
        """

        norm = 2*max(lta_truth)
        nare =  1/norm*torch.trapz(abs(lta_truth - lta_test), x=self.vertical_coord)   

        return nare
                                 
    #----------------------------------------------------------
    def reconstruct(self,y,model=None,undo_minmax=True,xmin=None,xmax=None):
        """
        Reconstructs flow using the decoder network. Reverses transform_minmax.
        
        INPUT:
            model       - AE model
            y           - latent space variables, shape (num_timesteps,encoding_dim)
            undo_minmax - whether to reverse min-max normalization
            xmin        - min. values of unnorm. data
            xmax        - max. values of unnorm. data
            
        RETURN:
            x - reconstructed flow, shape (num_timesteps,nfields,ny,nx)
        """
        
        if model is None:
            model = self.read_model()
            
        #if model was trained on PyTorch < v1.13.1
        for module in model.decoder:
            if isinstance(module,torch.nn.modules.upsampling.Upsample):
                module.recompute_scale_factor = None
        
        with torch.no_grad():
            model.eval()
            decoded = model.decoder(y)

        decoded = decoded.detach()

        if undo_minmax:
            decoded = self.unddo_minmax_transform(decoded,x_min=xmin,x_max=xmax)
            
        return decoded
        