The codes presented in this repository are developed to 
produce a mask that represent pixel dominated by signals 
from galaxies above a certain threshould pixel level.

A newly proposed method to define the threshould level uses 
the Gaussian Mixture Model (GMM) with the assumption that 
thre most dominant mixture component corresponds to the distribution 
of not source but background pixels.

A final object mask is defined to be the union of masks-per-band (i.e., 
the outcome of OR operation among the masks-per-band).

The proposed method is implemented for the Pan-STARRS1 stack images 
(https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images), and 
it also assumes that a target galaxy is centered in given images.

A relevant journal paper will be posted soon in the preprint server.

The script run_single_object.sh will help you figure out how the relevant 
Python programs work together to produce the object mask.
