import torch 
import torch.nn as nn
from torch.nn import (Conv2d, ConvTranspose2d, Dropout, Linear, MaxPool2d, Upsample, Flatten, BatchNorm1d, BatchNorm2d, InstanceNorm2d)
import torch.nn.functional as F
import sys
import numpy as np


""" 
Autoencoder network with four convolutional blocks in the Encoder (Down) and Decoder (Up) network.
"""



#-------------------------------------------------------
# I. Custome Layers
#-------------------------------------------------------
class Down(nn.Module):
    """
    Downsampling layer with mulitple convolutions.
    N x (Conv2d -> Activation) -> Downsample -> Normalization
    """
    def __init__(self,
                 channel_in,
                 channel_out,
                 activation=nn.ReLU(),
                 kernel_size=3,
                 kernel_stride=1,
                 downsample_kernel=2,
                 conv_padding='same',
                 padding_mode='zeros',
                 num_conv=2,
                 use_instance_norm=False,
                 downsample=True):
        
        super().__init__()
        self.channel_in        = channel_in
        self.channel_out       = channel_out
        self.kernel_size       = kernel_size
        self.kernel_stride     = kernel_stride
        self.conv_padding      = conv_padding
        self.padding_mode      = padding_mode
        self.activation        = activation
        self.downsample_kernel = downsample_kernel
        self.downsample        = downsample
        self.num_conv          = num_conv
        self.use_instance_norm = use_instance_norm
        
        if self.downsample:
            downsampling = MaxPool2d(kernel_size=self.downsample_kernel)
        else:
            downsampling = nn.Identity()
        
        if self.use_instance_norm:
            normalization = nn.InstanceNorm2d(self.channel_out)
        else:
            normalization = nn.BatchNorm2d(self.channel_out)        

        # Initialize layers
        #--------------------
        layers = list()
        for ii in range(self.num_conv):
            if ii < self.num_conv-1:
                layers.append(Conv2d(in_channels=self.channel_in,out_channels=self.channel_in,kernel_size=self.kernel_size,stride=self.kernel_stride,padding=self.conv_padding,padding_mode=self.padding_mode))
            else:
                layers.append(Conv2d(in_channels=self.channel_in,out_channels=self.channel_out,kernel_size=self.kernel_size,stride=self.kernel_stride,padding=self.conv_padding,padding_mode=self.padding_mode))    
            layers.append(self.activation)
        
        layers.append(downsampling)
        layers.append(normalization)
        self.layers = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.layers(x)

#-------------------------------------------------------
class Up(nn.Module):
    """
    N x(Conv2dTranspose -> Activation) -> Upsample -> Normalization 
    """
    def __init__(self,
                 channel_in,
                 channel_out,
                 activation=nn.ReLU(),
                 kernel_size=3,
                 kernel_stride=1,
                 upsampling_kernel=(2,2),
                 conv_padding=1,
                 num_conv=2,
                 use_instance_norm=False,
                 upsample=True):
        
        super().__init__()
        self.channel_in      = channel_in
        self.channel_out     = channel_out
        self.kernel_size     = kernel_size
        self.kernel_stride   = kernel_stride
        self.conv_padding    = conv_padding
        self.activation      = activation
        self.upsample        = upsample
        self.num_conv        = num_conv
        self.use_instance_norm = use_instance_norm
        
        if self.upsample:
            upsampling = Upsample(scale_factor=upsampling_kernel)
        else:
            upsampling = nn.Identity()
            
        if self.use_instance_norm:
            normalization = nn.InstanceNorm2d(self.channel_out)
        else:
            normalization = nn.BatchNorm2d(self.channel_out)
            
        # Initialize layers
        #--------------------
        layers = list()
        
        for ii in range(num_conv):
            if ii < self.num_conv-1:
                layers.append(ConvTranspose2d(in_channels=self.channel_in, out_channels=self.channel_in, kernel_size=self.kernel_size, stride=self.kernel_stride, padding=self.conv_padding))
            else:
                layers.append(ConvTranspose2d(in_channels=self.channel_in, out_channels=self.channel_out, kernel_size=self.kernel_size, stride=self.kernel_stride, padding=self.conv_padding))  
            layers.append(self.activation)
            
        layers.append(upsampling)
        layers.append(normalization)
        self.layers = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.layers(x)

#-------------------------------------------------------
class View(nn.Module):
    ''' Reshapes input tensor.'''
    def __init__(self, target_shape):
        super().__init__()
        self.shape = target_shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, inputs):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = inputs.size(0)
        shape = (batch_size, *self.shape)
        out = inputs.view(shape)
        return out

    

#-------------------------------------------------------
# II. AUTOENCODER
#-------------------------------------------------------
class Autoencoder(nn.Module):
    ''' Convolutional Autoencoder Model.'''

    def __init__(self, config=None):
        super(Autoencoder, self).__init__()
        
        if config is None:
            config = dict()
            
            
        
        # Data Parameters
        #-------------------
        self.nx,self.ny = config.get('nx',128),config.get('ny',32)
        self.nfields     = len(config.get('fields_use',[0,1,2]))

        # NN Setup
        #---------------------
        self.encoding_dim = config.get('encoding_dim',32)
        self.num_conv = config.get('num_conv',2)
        self.use_instance_norm = config.get('use_instance_norm',False)
        self.kernel_size = (3,3)
        self.kernel_stride = (1,1)
        
        # Dropout probabilities
        #------------------------
        self.p_lin = config.get('p_lin',0.15)
        self.p_conv = config.get('p_conv',0.15)
        
        
        # Activation functions
        #----------------------
        activations = {'sigmoid': nn.Sigmoid, 
                        'tanh':   nn.Tanh, 
                        'relu':   nn.ReLU,
                        'prelu':  nn.PReLU, 
                        'elu':    nn.ELU,
                        'selu':   nn.SELU,
                        'celu':   nn.CELU, 
                        'hardswish': nn.Hardswish,
                        'leakyrelu': nn.LeakyReLU}
        self.latent_activation = activations[config.get('latent_activation','sigmoid')]
        self.activation = activations[config.get('activation','relu')]
        
        
        # Channel sizes
        #---------------
        self.size_factor  = config.get('size_factor',4)
        self.channel_list = config.get('channel_list',[ int(self.size_factor*2**ii) for ii in range(4)])
        self.channel_list = [self.nfields,] + self.channel_list
        
        latent_shape = (self.channel_list[-1], self.ny//2**0, self.nx//2**4)
        flat_dim = np.prod(latent_shape)

        
        # Encoder Network
        #--------------
        self.encoder = nn.Sequential(
            
            Down(self.channel_list[0],self.channel_list[1],
                self.activation(),self.kernel_size,
                self.kernel_stride,downsample_kernel=(1,2),
                num_conv=self.num_conv,use_instance_norm=self.use_instance_norm,
                downsample=True),
            Dropout(p=self.p_conv),

            Down(self.channel_list[1],self.channel_list[2],
                self.activation(),self.kernel_size,
                self.kernel_stride,downsample_kernel=(1,2),
                num_conv=self.num_conv,use_instance_norm=self.use_instance_norm,
                downsample=True),
            Dropout(p=self.p_conv),

            Down(self.channel_list[2],self.channel_list[3],
                self.activation(),self.kernel_size,
                self.kernel_stride,downsample_kernel=(1,2),
                num_conv=self.num_conv,use_instance_norm=self.use_instance_norm,
                downsample=True),
            Dropout(p=self.p_conv),
            
            Down(self.channel_list[3],self.channel_list[4],
                self.activation(),self.kernel_size,
                self.kernel_stride,downsample_kernel=(1,2),
                num_conv=self.num_conv,use_instance_norm=self.use_instance_norm,
                downsample=True),
            Dropout(p=self.p_conv),
            
            Flatten(),
            
            Linear(in_features = flat_dim, out_features = self.encoding_dim),
            BatchNorm1d(self.encoding_dim),
            Dropout(p=self.p_lin),

            self.latent_activation(),
        )  

        
        # Decoder Network
        #----------------
        self.decoder = nn.Sequential(
            
            Linear(in_features = self.encoding_dim, out_features = flat_dim),
            BatchNorm1d(flat_dim),
            self.activation(),
            Dropout(p=self.p_lin),
            
            View(target_shape=latent_shape),
            
            Up(self.channel_list[4],self.channel_list[3],
               self.activation(),self.kernel_size, upsampling_kernel=(1,2),
               num_conv=self.num_conv, use_instance_norm=self.use_instance_norm,
               conv_padding=(self.kernel_size[0]//2,self.kernel_size[1]//2),upsample=True),
            Dropout(p=self.p_conv),
            
            Up(self.channel_list[3],self.channel_list[2],
               self.activation(),self.kernel_size, upsampling_kernel=(1,2),
               num_conv=self.num_conv, use_instance_norm=self.use_instance_norm,
               conv_padding=(self.kernel_size[0]//2,self.kernel_size[1]//2),upsample=True),
            Dropout(p=self.p_conv),
            
            Up(self.channel_list[2],self.channel_list[1],
               self.activation(),self.kernel_size, upsampling_kernel=(1,2),
               num_conv=self.num_conv, use_instance_norm=self.use_instance_norm,
               conv_padding=(self.kernel_size[0]//2,self.kernel_size[1]//2),upsample=True),
            Dropout(p=self.p_conv),
            
            Up(self.channel_list[1],self.channel_list[0],
               self.activation(),self.kernel_size, upsampling_kernel=(1,2),
               num_conv=self.num_conv, use_instance_norm=self.use_instance_norm,
               conv_padding=(self.kernel_size[0]//2,self.kernel_size[1]//2),upsample=True),
            Dropout(p=self.p_conv),
            
            ConvTranspose2d(in_channels=self.channel_list[0], 
                            out_channels=self.channel_list[0], 
                            kernel_size=self.kernel_size, 
                            stride=(1,1), 
                            padding=1),
            
            nn.Sigmoid(),
            
        )
        
            
    def forward(self, x):

        # Run AE
        #-------------------
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded, encoded
