import time
import numpy as np
from DcTNN.tnn import * 
from dc.dc import *
from phantominator import shepp_logan
from einops import rearrange
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def fft_2d(input, norm='ortho', dim=(-2, -1)):
    x = input
    x = rearrange(x, 'b c h w -> b h w c').contiguous()
    if x.shape[3] == 1:
        x = torch.cat([x, torch.zeros_like(x)], 3)
    x = torch.view_as_complex(x)
    x = torch.fft.fft2(x, norm=norm, dim=dim)
    x = torch.view_as_real(x)
    x = rearrange(x, 'b h w c -> b c h w').contiguous()
    return x

def ifft_2d(input, norm='ortho', dim=(-2, -1)):
    x = input
    x = rearrange(x, 'b c h w -> b h w c').contiguous()
    x = torch.view_as_complex(x)
    x = torch.fft.ifft2(x, dim=dim, norm=norm)
    x = torch.view_as_real(x)
    x = rearrange(x, 'b h w c -> b c h w').contiguous()
    return x

norm = 'ortho'
N = 320
R = 4
numCh = 1
iterations = 1
lamb = True

# Generate phantom for testing
ph = np.rot90(np.transpose(np.array(shepp_logan(N))), 1)
# As pytorch tensor
ph = torch.tensor(ph.copy(), dtype=torch.float)
# Create channel dim (with dummy data for the channel dimension)
ph = torch.unsqueeze(ph, 0)
ph = torch.cat([ph]*numCh, 0)
# Create batch dim
ph = torch.unsqueeze(ph, 0)

# Load a random sampling mask and inverse fourier shift
sampling_mask = np.array(ImageOps.grayscale(Image.open("masks/mask_R" + str(R) + ".png")))
sampling_mask = np.fft.ifftshift(sampling_mask) // np.max(np.abs(sampling_mask))
sampling_mask = torch.tensor(sampling_mask.copy(), dtype=torch.float)

# Undersample the image
y = fft_2d(ph) * sampling_mask
zf_image = ifft_2d(y, norm=norm)[:, 0:numCh, :, :]

# Define transformer encoder parameters
patchSize = 16
nhead_patch = 8
nhead_axial = 8
layerNo = 1
d_model_axial = None
d_model_patch = None
num_encoder_layers = 2
numCh = numCh
dim_feedforward = None

# Define the dictionaries of parameter values
patchArgs = {"patch_size": patchSize, "kaleidoscope": False, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
kdArgs = {"patch_size": patchSize, "kaleidoscope": True, "layerNo": layerNo, "numCh": numCh, "nhead": nhead_patch, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward, "d_model": d_model_patch}
axArgs = {"layerNo": layerNo, "numCh": numCh, "d_model": d_model_axial, "nhead": nhead_axial, "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward}

# Define the array of encoders
encList = [axVIT, patchVIT, patchVIT]
# Define the array of dictionaries
encArgs = [axArgs, kdArgs, patchArgs]

# Define the model
dcenc = cascadeNet(N, encList, encArgs, FFT_DC, lamb)

# Count the number of parameters
pytorch_total_params = sum(p.numel() for p in dcenc.parameters() if p.requires_grad)
print("Number of trainable params: " + str(pytorch_total_params))

# Perform model prediction
with torch.no_grad():
    start = time.time()
    phRecon = dcenc(zf_image, y, sampling_mask)
    end = time.time()
    elapsed = end - start
    print("Execution took " + str(elapsed) + " seconds")

# Illustrate operations
plt.close()
plt.figure(1)
plt.imshow(np.abs(ph[0, 0, :, :]))
plt.title("Phantom")

plt.figure(2)
plt.imshow(np.abs(zf_image[0, 0, :, :]))
plt.title("Zero Fill Image")

plt.figure(3)
plt.imshow(np.abs(phRecon[0, 0, :, :]))
plt.title("Reconstruction")

plt.figure(4)
plt.imshow(sampling_mask)
plt.title("Sampling Mask")
plt.show()

