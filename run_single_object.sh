#!/bin/bash

# Application of the Gaussian Mixture Model 
# to find the right threshould level for segmentation + 
# finding the target in the center of a given image + 
# producing a single object mask for the target.

# Developed by Min-Su Shin

# Script to demonstrate the procedure to generate a galaxy mask
# with PS1 stacked images in the g, r, i, z, and y bands.

OBJID="GLADE10001"

# image_converter: converting FITS files to NPY files.

for BAND in g r i z y
do
	./image_converter.py ./FITS_files/${OBJID}-${BAND}.fits ./output/Converted_Image_npy/${BAND}/${OBJID}-${BAND}.npy
done

# mask_generator: generating masks per band

for BAND in g r i z y
do
	USEPREFN="${OBJID}-${BAND}"
	./mask_generator.py ./FITS_files/${OBJID}-${BAND}.fits ./output/Generated_Mask_npy/${BAND} ${USEPREFN}
done

#  produce_single_object_mask: OR operation of the mask-per-bans npy files

SINGLEMASKDIR="./output/Single_Mask_npy"
INMASKDIR="./output/Generated_Mask_npy"
INIMAGEDIR="./output/Converted_Image_npy"
OUTIMAGEDIR="./output/Image_npy"
OUTMASKEDIMAGEDIR="./output/Masked_Image_npy"
OUTNANMASKDIR="./output/Nan_Mask_npy"
OUTGALAXYPLOTDIR="./output/Color_Image_png"

./produce_single_object_mask.py ${OBJID} ${SINGLEMASKDIR}/${OBJID}-object_mask.npy ${INMASKDIR} ${INIMAGEDIR} ${OUTIMAGEDIR} ${OUTMASKEDIMAGEDIR} ${OUTNANMASKDIR} ${OUTGALAXYPLOTDIR}/${OBJID}-yig.png

# mask_checker: check the object mask

./mask_checker.py ./output/Single_Mask_npy/${OBJID}-object_mask.npy ./output/Single_Mask_npy/${OBJID}-object_mask-check.log
