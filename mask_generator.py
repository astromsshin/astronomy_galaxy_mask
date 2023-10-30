#!/usr/bin/env python3

# Application of the Gaussian Mixture Model 
# to find the right threshould level for segmentation + 
# finding the target in the center of a given image + 
# producing a single object mask for the target.

# Developed by Min-Su Shin


import sys, math
import configparser

from tqdm import tqdm
from astropy.io import fits
from sklearn import mixture
from skimage import measure
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import numpy

import matplotlib
matplotlib.use('agg')

cfg_fn = "conf_mask_generator.ini"

try:
    input_fits_fn = sys.argv[1]
    output_directory = sys.argv[2]
    output_prefix = sys.argv[3]
except:
    print("(usage) %s (input FITS filename) (output directory) (output prefix)" % (sys.argv[0]))
    sys.exit(1)

# some parameters are loaded in the configuration file.
use_config = configparser.ConfigParser()
try:
    use_config.read(cfg_fn)
except:
    print("[ERROR] %s file is not found locally." % (cfg_fn))
    sys.exit(1)

log_fn = output_directory+"/"+output_prefix+'.log'

max_n_comp = use_config['GMM'].getint('max_n_comp')
max_iter_gmm = use_config['GMM'].getint('max_iter_gmm')
tol_gmm = use_config['GMM'].getfloat('tol_gmm')
min_top_weight = use_config['GMM'].getfloat('min_top_weight')

ratio_cut = use_config['Model'].getfloat('ratio_cut')
range_cut_min = use_config['Model'].getfloat('range_cut_min')
range_cut_max = use_config['Model'].getfloat('range_cut_max')
num_sample_x = use_config['Model'].getint('num_sample_x')
sigma_factor = use_config['Model'].getfloat('sigma_factor')
prob_cut_factor = use_config['Model'].getfloat('prob_cut_factor')
min_num_label_region = use_config['Model'].getfloat('min_num_label_region')

num_hist_bins = use_config['Plot'].getint('num_hist_bins')
sigma_factor_list = [float(e.strip()) for e in use_config['Plot']['sigma_factor_list'].split(',')]
prob_cut_factor_list = [float(e.strip()) for e in use_config['Plot']['prob_cut_factor_list'].split(',')]
flag_save_model_comparison = use_config['Plot'].getboolean('save_model_comparison')
flag_save_mixture_distribution = use_config['Plot'].getboolean('save_mixture_distribution')
flag_save_model_cut_results = use_config['Plot'].getboolean('save_model_cut_results')

log_fd = open(log_fn, 'w')

log_fd.write("[%s]\n" % (input_fits_fn))
try:
    print("Opening %s" % (input_fits_fn))
    hdulist = fits.open(input_fits_fn)
    img_header = hdulist[0].header
    img_data = hdulist[0].data
    hdulist.close()
except:
    print("[ERROR] reading %s failed." % (input_fits_fn))
    sys.exit(1)
width=img_data.shape[0]
height=img_data.shape[1]
img_data_1d = img_data.reshape(-1, 1)
num_pixels = width * height
print("... the number of pixels: %d" % (num_pixels))
print("... min. and max. of pixel values: %.6f %.6f" % (numpy.min(img_data_1d), \
numpy.max(img_data_1d)))
print("... w/o NaNs min. and max. of pixel values: %.6f %.6f" % (numpy.nanmin(img_data_1d), \
numpy.nanmax(img_data_1d)))
img_data_1d_for_gmm = img_data_1d[numpy.isfinite(img_data_1d)].reshape(-1,1)

bic_list = []
aic_list = []
aicc_list = []
model_list = []
print('### Estimating the multiple GMMs')
for n_comp in tqdm(range(1, max_n_comp+1)):
    gmm = mixture.GaussianMixture(n_components = n_comp, 
    covariance_type = 'full', tol = tol_gmm, max_iter = max_iter_gmm)
    model = gmm.fit(img_data_1d_for_gmm)
    model_list.append(model)
    bic_list.append(gmm.bic(img_data_1d_for_gmm))
    aic_list.append(gmm.aic(img_data_1d_for_gmm))
    aicc = gmm.aic(img_data_1d_for_gmm) + \
    2.0 * gmm._n_parameters() * (gmm._n_parameters() + 1.0) / (num_pixels - \
    gmm._n_parameters() - 1.0)
    aicc_list.append(aicc)

#print('BIC: ', bic_list)
#print('AIC: ', aic_list)
#print('AICc: ', aicc_list)
BIC_str = "BIC: "
AIC_str = "AIC: "
AICc_str = "AICc: "
for ind in range(0, len(bic_list)):
    BIC_str = BIC_str + ("%.6f" % (bic_list[ind])) + " "
    AIC_str = AIC_str + ("%.6f" % (aic_list[ind])) + " "
    AICc_str = AICc_str + ("%.6f" % (aicc_list[ind])) + " "
BIC_str = BIC_str.strip()
AIC_str = AIC_str.strip()
AICc_str = AICc_str.strip()
print("... " + BIC_str)
print("... " + AIC_str)
print("... " + AICc_str)
log_fd.write(BIC_str+"\n")
log_fd.write(AIC_str+"\n")
log_fd.write(AICc_str+"\n")

if flag_save_model_comparison:
    print("### Plotting the GMM model comparison results")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_n_comp+1), bic_list, label='BIC')
    plt.plot(range(1, max_n_comp+1), aic_list, label='AIC')
    plt.plot(range(1, max_n_comp+1), aicc_list, label='AICc')
    plt.xlabel("Number of components")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(output_directory+"/"+output_prefix+'-mixture_model_comparison.png')
    plt.close()

change_ratio = None
prev_val = None
best_n_comp = None
best_val = None
print("### Finding the best GMM with AICc")
for ind, val in enumerate(aicc_list):
    if ind == 0:
        change_ratio = 1.0
        prev_val = val
        best_val = val
        best_n_comp = 1
    else:
        if val <= prev_val:
            best_n_comp = ind + 1
            best_val = val
            ratio = abs((val - prev_val)/prev_val)
            if ratio <= ratio_cut:
                break
            else:
                prev_val = val
        else:
            break
print("... best_n_comp: ", best_n_comp, " with criteria val: ", best_val)
log_fd.write("best_n_comp: %d best_val: %.6f\n" % (best_n_comp, best_val))

best_model = model_list[best_n_comp-1]
percent_values = numpy.percentile(img_data_1d_for_gmm, [range_cut_min, range_cut_max])
test_x = numpy.linspace(percent_values[0], percent_values[1], num_sample_x)
logprob = best_model.score_samples(test_x.reshape(-1, 1))
responsibilities = best_model.predict_proba(test_x.reshape(-1, 1))
pdf = numpy.exp(logprob)
pdf_individual = responsibilities * pdf[:, numpy.newaxis]
if not best_model.converged_ :
    print("[PROBLEM] ... however, not converged.")
    sys.exit(1)
pdf_comp_weights = best_model.weights_
pdf_comp_means = best_model.means_
pdf_comp_covariances = best_model.covariances_
print('... weights: ', pdf_comp_weights)
pdf_comp_weights_str = ""
for weight in pdf_comp_weights:
    pdf_comp_weights_str = pdf_comp_weights_str + ("%.6f," % (weight))
pdf_comp_weights_str = pdf_comp_weights_str.strip(",")
log_fd.write("weights: " + pdf_comp_weights_str + "\n")
if max(pdf_comp_weights) < min_top_weight:
    log_fd.write("[WARNING] top weight < %.4f\n" % (min_top_weight))
    print("[WARNING] top weight < %.4f\n" % (min_top_weight))
#print('# means: ', pdf_comp_means)
#print('# covariances: ', pdf_comp_covariances)
dominant_comp_ind = numpy.argmax(pdf_comp_weights)
use_mean = pdf_comp_means[dominant_comp_ind].flatten()[0]
use_std_gmm = math.sqrt(pdf_comp_covariances[dominant_comp_ind].flatten()[0])
use_cut_gmm = use_mean + use_std_gmm * sigma_factor
minus_use_cut_gmm = use_mean - use_std_gmm * sigma_factor
data_lower_mean = img_data_1d_for_gmm[img_data_1d_for_gmm <= use_mean]
use_std_acc = use_mean - numpy.percentile(data_lower_mean, (50.0-34.13447460685)*2.0)
use_cut_acc =  use_mean + (use_mean - numpy.percentile(data_lower_mean, (100.0-prob_cut_factor)*2.00))
minus_use_cut_acc =  use_mean - (use_mean - numpy.percentile(data_lower_mean, (100.0-prob_cut_factor)*2.0))
print("[ADOPTED PARAMETER] %.6f %.6f (%.6f) for %s --> cut_gmm %.6f vs. cut_acc %.6f" % (use_mean, use_std_acc, use_std_gmm, input_fits_fn, use_cut_gmm, use_cut_acc))
log_fd.write("[ADOPTED PARAMETER: mean and std] %.6f %.6f (%.6f) --> cut_gmm %.6f vs. cut_acc %.6f\n" % (use_mean, use_std_acc, use_std_gmm, use_cut_gmm, use_cut_acc))


if flag_save_mixture_distribution:
    print("### Plotting the best GMM distribution")
    plt.figure(figsize=(8, 6))
    plt.hist(img_data_1d, bins=num_hist_bins, range=percent_values, density=True, histtype='stepfilled', alpha=0.4)
    plt.plot(test_x, pdf, '-k')
    plt.plot(test_x, pdf_individual, '--k')
    # plot mean and 1 sigma and 3 sigma and 5 sigma
    plt.axvline(use_mean, color='red')
    plt.axvline(use_mean+1.0*use_std_gmm, color='yellow', linestyle='dashed')
#    plt.axvline(use_mean+2.0*use_std_gmm, color='green', linestyle='dashed')
#    plt.axvline(use_mean+3.0*use_std_gmm, color='blue', linestyle='dashed')
    plt.axvline(use_cut_gmm, color='black', linestyle='dashed')
    plt.axvline(minus_use_cut_gmm, color='black', linestyle='dashed')
    plt.axvline(use_mean+1.0*use_std_acc, color='yellow', linestyle='solid')
#    plt.axvline(use_mean+2.0*use_std_acc, color='green', linestyle='solid')
#    plt.axvline(use_mean+3.0*use_std_acc, color='blue', linestyle='solid')
    plt.axvline(use_cut_acc, color='black', linestyle='solid')
    plt.axvline(minus_use_cut_acc, color='black', linestyle='solid')
    plt.title('Mean + 1 sigma & cut/-cut')
    plt.xlabel('Pixel value')
    plt.ylabel('Probability density')
    plt.savefig(output_directory+"/"+output_prefix+'-mixture_model_distribution.png')
    plt.close()

if flag_save_model_cut_results:
    print("### Plotting the model cut results")
    plt.figure(figsize=(12, 6))
    for ind in range(0, 3):
        cut_val = use_mean + (use_mean - numpy.percentile(data_lower_mean, (100.0-prob_cut_factor_list[ind])*2.0))
        plt.subplot(1,3,ind+1)
        temp_binary_result = img_data > cut_val
        plt.title("%.1f percent" % (prob_cut_factor_list[ind]))
        plt.imshow(temp_binary_result, cmap=plt.cm.gray)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_directory+"/"+output_prefix+'-acc-model_cut_results.png')
    plt.close()
    plt.figure(figsize=(12, 6))
    for ind in range(0, 3):
        cut_val = use_mean + use_std_gmm * sigma_factor_list[ind]
        plt.subplot(1,3,ind+1)
        temp_binary_result = img_data > cut_val
        plt.title("%.1f sigma" % (sigma_factor_list[ind]))
        plt.imshow(temp_binary_result, cmap=plt.cm.gray)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_directory+"/"+output_prefix+'-gmm-model_cut_results.png')
    plt.close()


########################## ACC
# produce a final mask
print("### Producing the best mask")
print("... with acc")
cut_val = use_cut_acc
binary_result = img_data > cut_val
use_label, num_labels = measure.label(binary_result, background=0, \
return_num=True)
max_label = numpy.max(use_label)
#... find the mean x, y for each label component
num_label_region_dict = dict()
x_label_region_dict = dict()
y_label_region_dict = dict()
mean_x_list = []
mean_y_list = []
mean_xy_distance_ratio_list = []
for ind in range(1, max_label+1):
    selected_region_ind = numpy.argwhere(use_label == ind)
    num_label_region_dict[ind] = selected_region_ind.shape[0]
    sum_x = 0.0
    sum_y = 0.0
    for region_ind in range(0, selected_region_ind.shape[0]):
        sum_x = sum_x + selected_region_ind[region_ind][1]
        sum_y = sum_y + selected_region_ind[region_ind][0]
    mean_x = sum_x / float(num_label_region_dict[ind])
    mean_y = sum_y / float(num_label_region_dict[ind])
    mean_x_list.append(mean_x)
    mean_y_list.append(mean_y)
    temp_disp_x = (mean_x/width - 0.5)
    temp_disp_y = (mean_y/height - 0.5)
    mean_xy_distance_ratio_list.append(temp_disp_x**2 + temp_disp_y**2)
#... find the central object and its label
best_ind = -1
best_pos_value = 1.0
best_label = None
for ind in range(1, max_label+1):
    if mean_xy_distance_ratio_list[ind - 1] < best_pos_value:
        best_pos_value = mean_xy_distance_ratio_list[ind - 1]
        best_ind = ind - 1
        best_label = ind
print("... best_label: %d with num_label_region: %d for best_pos_value: %.6f" % \
(best_label, num_label_region_dict[best_label], best_pos_value))
if num_label_region_dict[best_label] < min_num_label_region:
    print('[WARNING] num_label_region: %d is smaller than %d. -> re-estimation!' % \
    (num_label_region_dict[best_label], min_num_label_region))
    best_ind = -1
    best_pos_value = 1.0
    best_label = None
    for ind in range(1, max_label+1):
        if (mean_xy_distance_ratio_list[ind - 1] < best_pos_value) and \
        (num_label_region_dict[ind] >= min_num_label_region):
            best_pos_value = mean_xy_distance_ratio_list[ind - 1]
            best_ind = ind - 1
            best_label = ind
    print("... best_label: %d with num_label_region: %d for best_pos_value: %.6f" % \
    (best_label, num_label_region_dict[best_label], best_pos_value))

print("### Saving the mask")
#... target mask
target_mask = numpy.zeros(img_data.shape, dtype=bool)
use_ind = numpy.argwhere(use_label == best_label)
#print("... use_ind.size: ", use_ind.size)
for region_ind in range(0, use_ind.shape[0]):
    target_mask[use_ind[region_ind][0], use_ind[region_ind][1]] = True
#... convex hull
convex_hull_result = convex_hull_image(target_mask, offset_coordinates=False, tolerance=1e-20)
numpy.save(output_directory+"/"+output_prefix+'-acc-convexhull.npy', convex_hull_result)
#... binary result
numpy.save(output_directory+"/"+output_prefix+'-acc-threshold.npy', binary_result)
# Nope! #... and condition
# Nope! #final_mask = numpy.logical_and(binary_result, convex_hull_result)
numpy.save(output_directory+"/"+output_prefix+'-acc-mask.npy', target_mask)
###
print("### Plotting the result")
plt.figure(figsize=(12,12))
# image
plt.subplot(2,2,1)
plt.title("Image")
plt.imshow(img_data, vmin=percent_values[0], vmax=percent_values[1], cmap=plt.cm.gray)
plt.axis('off')
# binary threshoulding result
plt.subplot(2,2,2)
plt.title("Binary")
plt.imshow(binary_result, cmap=plt.cm.gray)
plt.axis('off')
# mask
plt.subplot(2,2,3)
plt.title("Mask")
plt.imshow(target_mask, cmap=plt.cm.gray)
#plt.scatter(mean_x_list, mean_y_list, c='red', marker='+')
plt.axis('off')
# convex hull result
plt.subplot(2,2,4)
plt.title("Convex hull")
plt.imshow(convex_hull_result, cmap=plt.cm.gray)
#plt.text(mean_x_list[best_ind], mean_y_list[best_ind], s='target', \
#fontsize=15, color='red', horizontalalignment='center', \
#verticalalignment='center')
plt.axis('off')
plt.tight_layout()
plt.savefig(output_directory+"/"+output_prefix+"-acc_cut_result.png")
plt.close()


########################## GMM
print("### Producing the best mask")
print("... with gmm")
cut_val = use_cut_gmm
binary_result = img_data > cut_val
use_label, num_labels = measure.label(binary_result, background=0, \
return_num=True)
max_label = numpy.max(use_label)
#... find the mean x, y for each label component
num_label_region_dict = dict()
x_label_region_dict = dict()
y_label_region_dict = dict()
mean_x_list = []
mean_y_list = []
mean_xy_distance_ratio_list = []
for ind in range(1, max_label+1):
    selected_region_ind = numpy.argwhere(use_label == ind)
    num_label_region_dict[ind] = selected_region_ind.shape[0]
    sum_x = 0.0
    sum_y = 0.0
    for region_ind in range(0, selected_region_ind.shape[0]):
        sum_x = sum_x + selected_region_ind[region_ind][1]
        sum_y = sum_y + selected_region_ind[region_ind][0]
    mean_x = sum_x / float(num_label_region_dict[ind])
    mean_y = sum_y / float(num_label_region_dict[ind])
    mean_x_list.append(mean_x)
    mean_y_list.append(mean_y)
    temp_disp_x = (mean_x/width - 0.5)
    temp_disp_y = (mean_y/height - 0.5)
    mean_xy_distance_ratio_list.append(temp_disp_x**2 + temp_disp_y**2)
#... find the central object and its label
best_ind = -1
best_pos_value = 1.0
best_label = None
for ind in range(1, max_label+1):
    if mean_xy_distance_ratio_list[ind - 1] < best_pos_value:
        best_pos_value = mean_xy_distance_ratio_list[ind - 1]
        best_ind = ind - 1
        best_label = ind
print("... best_label: %d with num_label_region: %d for best_pos_value: %.6f" % \
(best_label, num_label_region_dict[best_label], best_pos_value))
if num_label_region_dict[best_label] < min_num_label_region:
    print('[WARNING] num_label_region: %d is smaller than %d. -> re-estimation!' % \
    (num_label_region_dict[best_label], min_num_label_region))
    best_ind = -1
    best_pos_value = 1.0
    best_label = None
    for ind in range(1, max_label+1):
        if (mean_xy_distance_ratio_list[ind - 1] < best_pos_value) and \
        (num_label_region_dict[ind] >= min_num_label_region):
            best_pos_value = mean_xy_distance_ratio_list[ind - 1]
            best_ind = ind - 1
            best_label = ind
    print("... best_label: %d with num_label_region: %d for best_pos_value: %.6f" % \
    (best_label, num_label_region_dict[best_label], best_pos_value))

print("### Saving the mask")
#... target mask
target_mask = numpy.zeros(img_data.shape, dtype=bool)
use_ind = numpy.argwhere(use_label == best_label)
#print("... use_ind.size: ", use_ind.size)
for region_ind in range(0, use_ind.shape[0]):
    target_mask[use_ind[region_ind][0], use_ind[region_ind][1]] = True
#... convex hull
convex_hull_result = convex_hull_image(target_mask, offset_coordinates=False, tolerance=1e-20)
numpy.save(output_directory+"/"+output_prefix+'-gmm-convexhull.npy', convex_hull_result)
#... binary result
numpy.save(output_directory+"/"+output_prefix+'-gmm-threshold.npy', binary_result)
# Nope! #... and condition
# Nope! #final_mask = numpy.logical_and(binary_result, convex_hull_result)
numpy.save(output_directory+"/"+output_prefix+'-gmm-mask.npy', target_mask)
###
print("### Plotting the result")
plt.figure(figsize=(12,12))
# image
plt.subplot(2,2,1)
plt.title("Image")
plt.imshow(img_data, vmin=percent_values[0], vmax=percent_values[1], cmap=plt.cm.gray)
plt.axis('off')
# binary threshoulding result
plt.subplot(2,2,2)
plt.title("Binary")
plt.imshow(binary_result, cmap=plt.cm.gray)
plt.axis('off')
# mask
plt.subplot(2,2,3)
plt.title("Mask")
plt.imshow(target_mask, cmap=plt.cm.gray)
#plt.scatter(mean_x_list, mean_y_list, c='red', marker='+')
plt.axis('off')
# convex hull result
plt.subplot(2,2,4)
plt.title("Convex hull")
plt.imshow(convex_hull_result, cmap=plt.cm.gray)
#plt.text(mean_x_list[best_ind], mean_y_list[best_ind], s='target', \
#fontsize=15, color='red', horizontalalignment='center', \
#verticalalignment='center')
plt.axis('off')
plt.tight_layout()
plt.savefig(output_directory+"/"+output_prefix+"-gmm_cut_result.png")
plt.close()

log_fd.close()
