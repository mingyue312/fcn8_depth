import numpy as np
import skimage
import skimage.io
import scipy.io as sio
import skimage.transform

# R range: [0, 65535]
# G range: [0, 65535]
# B range: [0, 65535]
# IR range: [0, 65535]
# depth range: [0,343.40332]
# [R_mean, G_mean, B_mean, IR_mean, Depth_mean]
VGG_MEAN = [31008.955/65535*255, 29200.743/65535*255, 27095.209/65535*255, 32855.788/65535*255, 73.876/343*255]

# this list of tuples reduces the number of unique semantic labels in the ground truth by merging.
GT_DICT = [(0, 0), #unmapped area
           (128, 1), #road
           (255, 2)] #building

def read_mat(path):
    return np.load(path)

def write_mat(path, m):
    np.save(path, m)

def read_ids(path):
    return [line.rstrip('\n') for line in open(path)]

def image_scaling(rgb_scaled):
    # scales input images by VGG mean since we are likely initializing with the VGG16 parameters for conv1-conv5
    # value/max*255 then subtract mean/max*255
    rgb_scaled = rgb_scaled.astype(float)
    rgb_scaled[:, :, 0] = rgb_scaled[:, :, 0]/65535*255 - VGG_MEAN[0]
    rgb_scaled[:, :, 1] = rgb_scaled[:, :, 1]/65535*255 - VGG_MEAN[1]
    rgb_scaled[:, :, 2] = rgb_scaled[:, :, 2]/65535*255 - VGG_MEAN[2]
    rgb_scaled[:, :, 3] = rgb_scaled[:, :, 3]/65535*255 - VGG_MEAN[3]
    rgb_scaled[:, :, 4] = rgb_scaled[:, :, 4]/343*255 - VGG_MEAN[4]

    return rgb_scaled

def GT_KEYS_to_index(array):
    for i in range(len(GT_DICT)):
        array[array == GT_DICT[i][0]] = GT_DICT[i][1]