#!/usr/bin/env python3

# Application of the Gaussian Mixture Model 
# to find the right threshould level for segmentation + 
# finding the target in the center of a given image + 
# producing a single object mask for the target.

# Developed by Min-Su Shin


import sys
import numpy
from astropy.io import fits

warning_fraction = 0.80

try:
    input_mask_fn = sys.argv[1]
    output_log_fn = sys.argv[2]
except:
    print("(usage) %s (input mask NPY filename) (output log filename)" % (sys.argv[0]))
    sys.exit(1)

try:
    print("Input: %s" % (input_mask_fn))
    mask_data = numpy.load(input_mask_fn)
except:
    print("[ERROR] reading %s failed." % (input_mask_fn))
    sys.exit(1)

try:
    print("Writing %s" % (output_log_fn))
    logfd = open(output_log_fn, 'w')
except:
    print("[ERROR] opening %s failed." % (output_log_fn))
    sys.exit(1)


mask_size_x, mask_size_y = mask_data.shape
out_str = "Mask size for %s: %d %d" % (input_mask_fn, mask_size_x, mask_size_y)
print(out_str)
true_cnt = numpy.count_nonzero(mask_data)
# Target size and its fraction
true_fraction = true_cnt/(mask_size_x*mask_size_y*1.0)
out_str = "target size and fraction: %d %.6f" % (true_cnt, true_fraction)
print(out_str)
logfd.write(out_str+"\n")
if true_fraction > warning_fraction:
    print("... WARNING %.6f > %.2f" % (true_fraction, warning_fraction))


# check whether the mask touches edges of the image.
check_xmin = numpy.any(mask_data[0,:])
check_xmax = numpy.any(mask_data[mask_size_x-1,:])
check_ymin = numpy.any(mask_data[:,0])
check_ymax = numpy.any(mask_data[:,mask_size_y-1])
out_str = "edge x: %s %s" % (check_xmin, check_xmax)
print(out_str)
logfd.write(out_str+"\n")
out_str = "edge y: %s %s" % (check_ymin, check_ymax)
print(out_str)
logfd.write(out_str+"\n")
if check_xmin or check_xmax or check_ymin or check_ymax:
    print("... WARNING %s" % (input_mask_fn))
