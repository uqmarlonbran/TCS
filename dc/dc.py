"""
Performs data consistency operations.

Author: Marlon Ernesto Bran Lorenzana
Date: August 19, 2021
"""
import torch

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
        z = (1 - mask) * z + mask * (z + lamb * y) / (1 + lamb)

    # Apply mask and invert (keep channels that we are working with)
    z = torch.view_as_real(torch.fft.ifft2(z, norm=norm))[:, :, :, 0:numCh]
    z = z.permute(0, 3, 1, 2)

    # Return masked image
    return z
