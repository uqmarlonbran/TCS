"""
Creates generic vision transformers.

Author: Marlon Ernesto Bran Lorenzana
Date: February 15, 2022
"""

import torch
from dc.dc import *
from einops.layers.torch import Rearrange
from torch import nn

# Helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class patchVIT(nn.Module):
    """
    Defines a TNN that creates either Kaleidoscope or Patch tokens.
    Args:
        N (int)                     -       Image Size
        patch_size (int)            -       Size of patch or Kaleidoscope tokens
        kaleidoscope (bool)         -       If true the network will create Kaleidoscope tokens
        layerNo (int)               -       Number of cascaded TNN
        numCh (int)                 -       Number of input channels (real, imag)
        d_model (int)               -       Model dimension
        nhead (int)                 -       Number of heads to use in multi-head attention
        num_encoder_layers (int)    -       Number of encoder layers within each encoder
        dim_feedforward (int)       -       Feedforward size in the MLP
        dropout (float)             -       Dropout of various layers
        activation                  -       Defines activation function for transformer encoder
    """
    def __init__(self, N, patch_size=16, kaleidoscope=False, layerNo=2, numCh=1, d_model=None, 
                    nhead=8, num_encoder_layers=2, dim_feedforward=None, dropout=0.1, activation='relu', 
                    layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None):
        super(patchVIT, self).__init__()
        # Define the number of iterations to go through the transformer
        self.layerNo = layerNo
        # Define the number of channels (2 means imaginary)
        self.numCh = numCh
        # Define the image size
        self.N = N
        # Define the patch size
        self.patch_size = patch_size
        
        # Determine d_model size
        if d_model is None:
            d_model = patch_size * patch_size * numCh
        # Determine dim_feedforward if not given
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))
        # Whether or not kaleidoscope
        self.kaleidoscope = kaleidoscope

        # For each layer, cascade an image transformer
        transformers = []
        for _ in range(layerNo):
            transformers.append(imageEncoder(self.N, self.patch_size, numCh, kaleidoscope,
                                                            d_model, nhead, num_encoder_layers, dim_feedforward,
                                                            dropout, activation, layer_norm_eps, 
                                                            batch_first, device, dtype))
        self.transformers = nn.ModuleList(transformers)
    
    """
    xPrev should be [numBatch, numCh, ydim, xdim]
    """
    def forward(self, xPrev):

        im = xPrev

        # Go over the number of iterations to perform updating 
        for i in range(self.layerNo):           

            # Get a new image estimate based on previous estimate of x (xPrev) 
            im_denoise = self.transformers[i](im)
            im = im_denoise

        # Return the final output and the residual 
        return im


class axVIT(nn.Module):
    """
    Defines the transformer for MRI reconstruction using exclusively a Transformer Encoder and axial tokens
    Args:
            N (int)                     -       Image Size
            layerNo (int)               -       Number of cascaded TNN
            numCh (int)                 -       Number of input channels (real, imag)
            d_model (int)               -       Model dimension
            nhead (int)                 -       Number of heads to use in multi-head attention
            num_encoder_layers (int)    -       Number of encoder layers within each encoder
            dim_feedforward (int)       -       Feedforward size in the MLP
            dropout (float)             -       Dropout of various layers
            activation                  -       Defines activation function for transformer encoder
    """
    def __init__(self, N, layerNo=2, numCh=1, d_model=None, nhead=8, num_encoder_layers=2,
                    dim_feedforward=None, dropout=0.1, activation='relu',
                    layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None):
        super(axVIT, self).__init__()
        # Define the number of iterations to go through the transformer - this could be projections I suppose
        self.layerNo = layerNo
        # Define the number of channels (2 means imaginary)
        self.numCh = numCh
        # Define the image size
        self.N = N
        
        # Determine d_model size, can't be a prime number if we want MH Attention
        if d_model is None:
            d_model = N * numCh
        # Determine dim_feedforward if not given
        if dim_feedforward is None:
            dim_feedforward = int(d_model ** (3 / 2))
        

        # Cascade the denoising transformers
        transformers = []
        for _ in range(layerNo):
            transformers.append(axialEncoder(self.N, numCh, d_model, nhead, num_encoder_layers, dim_feedforward,
                                        dropout, activation, layer_norm_eps, batch_first, device, dtype))
        self.transformers = nn.ModuleList(transformers)
    
    """
    xPrev should be [numBatch, numCh, ydim, xdim]
    """
    def forward(self, xPrev):

        im = xPrev

        # Go over the number of iterations to perform updating 
        for i in range(self.layerNo):           

            # Get a new image estimate based on previous estimate of x (xPrev) 
            im_denoise = self.transformers[i](im)
            im = im_denoise

        # Return the final output and the residual 
        return im


class cascadeNet(nn.Module):
    """
    Defines a TNN that cascades denoising networks and applies data consistency.
    Args:
        N (int)                     -       Image Size
        encList (array)             -       Should contain denoising network
        encArgs (array)             -       Contains dictionaries with args for encoders in encList
        dcFunc (function)           -       Contains the data consistency function to be used in recon
        lamb (bool)                 -       Whether or not to use a leanred data consistency parameter
    """
    def __init__(self, N, encList, encArgs, dcFunc=FFT_DC, lamb=True):
        super(cascadeNet, self).__init__()
        # Define lambda for data consistency
        if lamb:
            self.lamb = nn.Parameter(torch.ones(len(encList)) * 0.5)
        else:
            self.lamb = False
        # Define image size
        self.N = N
        # Define the data consistency function
        self.dcFunc = dcFunc

        # Cascade the transformers
        transformers = []
        for i, enc in enumerate(encList):
            transformers.append(enc(N, **encArgs[i]))

        self.transformers = nn.ModuleList(transformers)

    """
    xPrev should be [numBatch, numCh, ydim, xdim]
    y should be [numBatch, kCh, ydim, xdim]
    sampleMask should be [ydim, xdim]
    """
    def forward(self, xPrev, y, sampleMask):

        im = xPrev

        # Go over the number of iterations to perform updating 
        for i, transformer in enumerate(self.transformers):        

            # Denoise the image
            im_denoise = transformer(im)
            im = im + im_denoise

            # Update the residual
            if self.lamb is False:
                im = self.dcFunc(im, y, sampleMask, None)
            else:
                im = self.dcFunc(im, y, sampleMask, self.lamb[i])

        # Return the final output
        return im


class imageEncoder(nn.Module):
    """
    Here we initialize a standard Encoder that utilizes image patches or kaleidoscope tokens.

    Args are the same as that defined by the normal Encoder class
    """
    def __init__(self, image_size, patch_size, numCh=1, kaleidoscope=False, d_model=512, nhead=8, 
                num_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', layer_norm_eps=1e-05,
                batch_first=True, device=None, dtype=None, norm=None):
        super().__init__()
        # Define the transformer
        self.encoderLayer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, layer_norm_eps, batch_first, device, dtype)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers, norm=norm)
        # Define size of the transformer 
        self.d_model = d_model

        ## Define the image size params
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # Get the patch dimensionality
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = patch_height * patch_width * numCh

        # Define Kaleidoscope transform
        if kaleidoscope:
            self.to_embedding = nn.Sequential(
                Rearrange('b c (k1 h) (k2 w) -> b (h w) (k1 k2 c)', k1=patch_height, k2=patch_width),
                nn.Linear(patch_dim, d_model)
            )
            self.from_embedding = Rearrange('b (h w) (k1 k2 c) -> b c (k1 h) (k2 w)', k1=patch_height, k2=patch_width, h=image_height // patch_height, c=numCh)
        else:
            # Embed the image in patches
            self.to_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(patch_dim, d_model),
            )
            self.from_embedding = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=numCh, h=image_height // patch_height, p1=patch_height, p2=patch_width)

        # Define positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Define layer normalisation and linear transformation. As well-as de-patching the image.
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_dim),
            self.from_embedding,
        )

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)


    # Functions as a wrapper for transformer function that first creates tokens from the image
    def forward(self, img, src_mask=None):

        x = img
        
        # Get the patch representation
        x = self.to_embedding(x)
        
        # Get the positional embedding
        x = x + self.pos_embedding

        # Perform dropout
        x = self.dropout(x)

        # Get the output of the transformer
        x = self.encoder(x, src_mask)

        # Pass-through multi-layer perceptron and un-patch
        x = self.mlp_head(x)      

        # Return the output
        return x


class axialEncoder(nn.Module):
    """
    Initializes a standard Encoder that utilizes axial attention.

    Args are the same as that defined by the normal Encoder class
    """
    def __init__(self, image_size, numCh=1, d_model=512, nhead=8, num_layers=6, dim_feedforward=None, dropout=0.1, 
                    activation='relu', layer_norm_eps=1e-05, batch_first=True, device=None, dtype=None, norm=None):
        super().__init__()
        # Define the transformer
        self.encoderLayer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, layer_norm_eps, batch_first, device, dtype)
        # Because we cascade veritcal and horizontal transformers,
        # they must share the total number of encoder layers
        numLayers = num_layers // 2
        if numLayers < 1:
            numLayers = 1
        # Define each of the encoders
        self.horizontalEncoder = nn.TransformerEncoder(self.encoderLayer, numLayers, norm=norm)
        self.verticalEncoder = nn.TransformerEncoder(self.encoderLayer, numLayers, norm=norm)
        
        # Define size of the transformer 
        self.d_model = d_model

        ## Define image size
        image_height, image_width = pair(image_size)

        # Embed the slices horizontally
        self.to_horizontal_embedding = nn.Sequential(
            Rearrange('b c h w -> b h (w c)'),
            nn.Linear(image_width * numCh, d_model)
        )

        # Embed the slices vertically
        self.to_vertical_embedding = nn.Sequential(
            Rearrange('b c h w -> b w (h c)'),
            nn.Linear(image_height * numCh, d_model)
        )

        # Define positional embedding
        self.horizontal_pos_embedding = nn.Parameter(torch.randn(1, image_width, d_model))
        self.vertical_pos_embedding = nn.Parameter(torch.randn(1, image_height, d_model))

        # Define layer normalisation and linear transformation.
        self.horizontal_mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, image_width * numCh),
            Rearrange('b h (w c) -> b c h w', c=numCh)
        )
        self.vertical_mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, image_height * numCh),
            Rearrange('b w (h c) -> b c h w', c=numCh)
        )

        # Define dropout layer
        self.dropout = nn.Dropout(dropout)



    # Functions as a wrapper for transformer function that first creates tokens from the image
    def forward(self, img, src_mask=None, src_key_padding_mask=None):

        x = img
        
        # Get the horizontal representation
        x = self.to_horizontal_embedding(x)
        
        # Get the positional embedding
        x = x + self.horizontal_pos_embedding

        # Perform dropout
        x = self.dropout(x)

        # Get the output of horizontal MHA
        x = self.horizontalEncoder(x, src_mask, src_key_padding_mask)

        # Pass-through multi-layer perceptron and un-token
        x = self.horizontal_mlp_head(x) 

        # Get the vertical representation
        x = self.to_vertical_embedding(x)
        
        # Get the positional embedding
        x = x + self.vertical_pos_embedding

        # Perform dropout
        x = self.dropout(x)

        # Get the output of horizontal MHA
        x = self.verticalEncoder(x, src_mask, src_key_padding_mask)

        # Pass-through multi-layer perceptron and un-token
        x = self.vertical_mlp_head(x)

        # Return the output
        return x

