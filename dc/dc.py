"""
Performs data consistency operations.

Author: Marlon Ernesto Bran Lorenzana
Date: August 19, 2021
"""
import torch
from einops import rearrange

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

"""
Applies Fourier mask and data consistency in Fourier Space
Inputs:
    x       -       Image estimate:
                        -   dimensions [minibatch, nchannels (real, imag), xdim, ydim]
    y       -       Collected k-space samples (complex valued):
                        -   dimensions [minibatch, nchannels (real, imag), xdim, ydim]
    mask    -       k-space mask:
                        -   dimensions [xdim, ydim]
    lamb    -       Lambda:
                        -   data consistency that is learned
Outputs:            z 
"""
def FFT_DC(x, y, mask, lamb, norm='ortho'):

    # Check if complex
    numCh = x.shape[1]

    # Get complex y view
    cy = y.permute(0, 2, 3, 1).contiguous()
    cy = torch.view_as_complex(cy)

    # By default, torch.view_as_complex uses last index as real,imag
    x = x.permute(0, 2, 3, 1).contiguous()

    # Perform operations depending on the number of dimensions
    if numCh == 1:

        # Populate imaginary axis with zeros
        x = torch.cat([x, torch.zeros_like(x)], 3)

    # get k-space of the input image
    z = torch.fft.fft2(torch.view_as_complex(x), norm=norm)

    # Perform data consistency 
    if lamb is None:
        # Replace Fourier measurements
        z = (1 - mask) * z + mask * y
    else:
        # Weighted average of the collected and reconstructed points
        z = (1 - mask) * z + mask * (z + lamb * cy) / (1 + lamb)

    # Apply mask and invert (keep channels that we are working with)
    z = torch.view_as_real(torch.fft.ifft2(z, norm=norm))[:, :, :, 0:numCh]
    z = z.permute(0, 3, 1, 2)

    # Return masked image
    return z
