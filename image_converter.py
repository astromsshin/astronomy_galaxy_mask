#!/usr/bin/env python3

# Application of the Gaussian Mixture Model 
# to find the right threshould level for segmentation + 
# finding the target in the center of a given image + 
# producing a single object mask for the target.

# Developed by Min-Su Shin

import sys

from astropy.io import fits
import numpy

try:
    input_fits_fn = sys.argv[1]
    output_npy_fn = sys.argv[2]
except:
    print("(usage) %s (input FITS filename) (output NPY filename)" % (sys.argv[0]))
    sys.exit(1)

try:
    hdulist = fits.open(input_fits_fn)
    img_header = hdulist[0].header
    img_data = hdulist[0].data
    hdulist.close()
except:
    print("[ERROR] reading %s failed." % (input_fits_fn))
    sys.exit(1)

width=img_data.shape[0]
height=img_data.shape[1]
print("... " + input_fits_fn, img_data.shape, img_data.dtype)
numpy.save(output_npy_fn, img_data)
