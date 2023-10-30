#!/usr/bin/env python3

# Application of the Gaussian Mixture Model 
# to find the right threshould level for segmentation + 
# finding the target in the center of a given image + 
# producing a single object mask for the target.

# Developed by Min-Su Shin


import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

import img_scale

default_nonlinearity = 5.0

### Some constants ###
# for 480 x 480 pixels
USEMINPIX=240
USEMAXPIX=USEMINPIX+480
MASKTYPE="acc"

band_list = ['g', 'r', 'i', 'z', 'y']
# RGB
plot_band = ['y', 'i', 'g']
#plot_band = ['i', 'r', 'g']

try:
    object_id = sys.argv[1]
    out_mask_fn = sys.argv[2]
    in_mask_directory = sys.argv[3]
    in_image_directory = sys.argv[4]
    out_image_directory = sys.argv[5]
    out_masked_image_directory = sys.argv[6]
    out_nan_mask_directory = sys.argv[7]
    out_galaxy_plot_fn = sys.argv[8]
except:
    print("Usage: %s (object id) (output mask filename) (input mask directory) (input image directory) (output image directory) (output masked image directory) (output nan-mask directory) (output clor image filename)" % (sys.argv[0]))
    sys.exit(1)

# load all input masks
in_mask_fn_list = []
for band in ["g", "r", "i", "z", "y"]:
    in_mask_fn = in_mask_directory + "/" + band + "/" + object_id + "-" + band + "-" + MASKTYPE + "-mask.npy"
    in_mask_fn_list.append(in_mask_fn)
in_mask_list = []
for single_in_mask_fn in in_mask_fn_list:
    in_mask_list.append(np.load(single_in_mask_fn))
print("... in_mask.shape:", in_mask_list[0].shape)

# or-condition combined mask
out_mask_large = in_mask_list[0]
for ind in range(1, len(band_list)):
    out_mask_large = out_mask_large | in_mask_list[ind]
print("# out_mask_fn: ", out_mask_fn)
out_mask = out_mask_large[USEMINPIX:USEMAXPIX,USEMINPIX:USEMAXPIX]
print("... out_mask.shape:", out_mask.shape)
np.save(out_mask_fn, out_mask)
# plot
print("# producing %s.png" % (out_mask_fn))
plt.figure(figsize=(6, 4))
plt.imshow(out_mask, vmin=0, vmax=1, filternorm=False, cmap=plt.cm.gray)
plt.axis('off')
plt.tight_layout()
plt.savefig(out_mask_fn+".png")
plt.close()


rgb_array = np.empty((out_mask.shape[0], out_mask.shape[1], 3), 
dtype=float)
rgb_masked_array = np.empty((out_mask.shape[0], out_mask.shape[1], 3), 
dtype=float)
# output image, masked image, and nan-mask directory
for band in ["g", "r", "i", "z", "y"]:
    in_image_fn = in_image_directory + "/" + band + "/" + object_id + "-" + band + ".npy"
    out_image_fn = out_image_directory + "/" + band + "/" + object_id + "-" + band + ".npy"
    out_masked_image_fn = out_masked_image_directory + "/" + band + "/" + object_id + "-" + band + "-masked.npy"
    out_nan_mask_fn = out_nan_mask_directory + "/" + band + "/" + object_id + "-" + band + "-nan.npy"
    print("# out_image_fn: ", out_image_fn)
    in_image = np.load(in_image_fn)
    print("... in_image.shape:", in_image.shape)
    out_image = in_image[USEMINPIX:USEMAXPIX,USEMINPIX:USEMAXPIX]
    print("... out_image.shape:", out_image.shape)
    np.save(out_image_fn, out_image)
    print("# out_masked_image_fn: ", out_masked_image_fn)
    out_masked_image = np.multiply(out_image, out_mask)
    try:
        found_index = plot_band.index(band)
        log_mask_fn = in_mask_directory + "/" + band + "/" + object_id + "-" + band + ".log"
        with open(log_mask_fn, 'r') as f:
            last_line = f.readlines()[-1]
        use_min_val = float(last_line.split()[5]) + 0.1
        temp_min_val, use_max_val = img_scale.range_from_percentile(out_masked_image[out_masked_image >= use_min_val], 
        low_cut=0.20, high_cut=0.30)
        rgb_array[:,:,found_index] = img_scale.asinh(out_image, 
        scale_min = temp_min_val, non_linear=default_nonlinearity)
        rgb_masked_array[:,:,found_index] = img_scale.asinh(out_masked_image, 
        scale_min = temp_min_val, non_linear=default_nonlinearity)
    except ValueError:
        pass
    print("... out_masked_image.shape:", out_masked_image.shape)
    np.save(out_masked_image_fn, out_masked_image)
    print("# out_nan_mask_fn: ", out_nan_mask_fn)
    masked_nan = np.ma.masked_invalid(out_masked_image)
    out_nan_mask = np.ma.getmask(masked_nan)
    print("... out_nan_mask.shape:", out_nan_mask.shape)
    np.save(out_nan_mask_fn, out_nan_mask)

# plot
print("# producing %s" % (out_galaxy_plot_fn))
plt.figure(figsize=(9, 4))
plt.subplot(1,2,1)
plt.imshow(rgb_array, interpolation='nearest')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(rgb_masked_array, interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.savefig(out_galaxy_plot_fn)
plt.close()
