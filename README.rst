TRANSFORMER COMPRESSED SENSING VIA GLOBAL IMAGE TOKENS

=========================================================================

This repository contains the supplementary material and PyTorch implementation of Deep cascade of Transformer Neural Networks (DcTNN). 

Environment Dependencies
====================

Code is known to work with:

* Python 3.8.10
* PyTorch 1.9.0
* Einops 0.3.0
* Pillow 8.0.0

----

Usage
====================

``build_model.py`` provides an example of how to build and test DcTNN, cascading Axial (Ax), Kaleidoscope (KD) and Patch (P) transformer encoder layers to build our Ensemble network.

Class ``cascadeNet`` takes the following arguments:


+---------------+-----------+------------------------------------------------------------------+
| Args          | Type      | Description                                                      |
+===============+===========+==================================================================+
| ``N``         | int       | Image Size                                                       |
+---------------+-----------+------------------------------------------------------------------+
| ``encList``   | array     | Is an array that contains denoising transformer encoder modules  |
+---------------+-----------+------------------------------------------------------------------+
| ``encArgs``   | array     | Contains dictionaries with args for encoders in encList          |
+---------------+-----------+------------------------------------------------------------------+
| ``dcFunc``    | function  | Contains the data consistency function to be used in recon       |
+---------------+-----------+------------------------------------------------------------------+
| ``lamb``      | bool      | Whether or not to use a leanred data consistency parameter       |
+---------------+-----------+------------------------------------------------------------------+

Classes ``axVIT`` and ``patchVIT`` construct image denoisers based on ``axialEncoder`` and ``imageEncoder`` respectively; ``patchVIT`` can be configured as a Patch or Kaleidoscope Encoder. 


Therefore to build the denoiser featured in our paper:

.. code-block:: python 
  
  # Image size
  N = 320
  # Size of patches/kd tokens
  patchSize = 16
  # Number of heads for patch/kd encoders
  nhead_patch = 8
  # Number of heads for axial encoder
  nhead_axial = 8
  # Number of cascaded denoising blocks within each TNN
  layerNo = 1
  # None d_model defaults to input dimension
  d_model_axial = None
  d_model_patch = None
  # Number of encoder layers per-transformer
  num_encoder_layers = 2
  # Number of channels of the image
  numCh = 1
  # None dim_feedforward defaults to d_model^(3/2)
  dim_feedforward = None

  # Define the dictionaries of parameter values
  patchArgs = {"patch_size": patchSize, "kaleidoscope": False, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
  kdArgs = {"patch_size": patchSize, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
  axArgs = {"layerNo": layerNo, "numCh": numCh, "d_model": d_model_axial, "nhead": nhead_axial, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward}
  
  encList = [axVIT, patchVIT, patchVIT]
  encArgs = [axArgs, kdArgs, patchArgs]
  dcFunc = FFT_DC
  lamb = True

  # Define the model
  dcenc = cascadeNet(N, encList, encArgs, dcFunc, lamb)
----

Citation and Acknowledgement
====================

Paper is available on arXiv::

  M. Bran Lorenzana, C. Engstrom, F. Liu, and S. S. Chandra, ‘Transformer Compressed Sensing via Global Image Tokens’. arXiv, Mar. 27, 2022. Available: http://arxiv.org/abs/2203.12861 


----