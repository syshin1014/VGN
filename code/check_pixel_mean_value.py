# To get the pixel mean value from training images 
# coded by syshin (170805)
# DRIVE : [ 180.73042427   97.27726367   57.10662087] on mask
# DRIVE : [ 126.83705873   69.0154593    41.42158474]
# STARE : [ 150.29591358   83.55034309   27.50114876]
# ALL : [ 136.00693286   74.69702627   35.98020038] # DRIVE+STARE
# CHASE_DB1 : [ 113.95299712  39.80676852   6.88010585]
# HRF : [164.41978937  51.82606062  27.12979025]


import os
import numpy as np
import skimage.io

from config import cfg


DATASET='HRF'


if DATASET=='DRIVE':
    
    dataset_root_path = '/mnt/hdd1/DRIVE'
    train_img_names = sorted(os.listdir(os.path.join(dataset_root_path, 'training/images')))
    train_img_names = map(lambda x: x[:2], train_img_names)
    rgb_cum_sum = np.zeros((3,))
    cum_num_pixels = 0.
    for cur_img_name in train_img_names:
        cur_img = skimage.io.imread(os.path.join(dataset_root_path,'training/images',cur_img_name+'_training.tif'))
        #cur_mask = skimage.io.imread(os.path.join(dataset_root_path,'training/mask',cur_img_name+'_training_mask.gif'))
        #cur_mask = cur_mask>100
        #cur_img = cur_img*np.dstack((cur_mask,cur_mask,cur_mask))
        cur_rgb_sum = np.sum(cur_img, axis=(0,1))
        rgb_cum_sum = rgb_cum_sum + cur_rgb_sum
        #cum_num_pixels += np.sum(cur_mask)
        cum_num_pixels += np.cumprod(cur_img.shape)[1]

    mean_rgb_val = rgb_cum_sum/cum_num_pixels    
    print mean_rgb_val

elif DATASET=='STARE':
    
    dataset_root_path = '/mnt/hdd1/STARE'
    train_img_names = sorted(os.listdir(os.path.join(dataset_root_path, 'stare-images')))
    train_img_names = map(lambda x: x[:6], train_img_names[:10])
    rgb_cum_sum = np.zeros((3,))
    cum_num_pixels = 0.
    for cur_img_name in train_img_names:
        cur_img = skimage.io.imread(os.path.join(dataset_root_path,'stare-images',cur_img_name+'.ppm'))
        cur_rgb_sum = np.sum(cur_img, axis=(0,1))
        rgb_cum_sum = rgb_cum_sum + cur_rgb_sum
        cum_num_pixels += np.cumprod(np.shape(cur_img))[1]

    mean_rgb_val = rgb_cum_sum/cum_num_pixels
    print mean_rgb_val
    
elif DATASET=='CHASE_DB1':
    
    train_set_txt_path = '/home/syshin/Documents/CA/CHASE_DB1/train.txt'

    with open(train_set_txt_path) as f:
        train_img_names = [x.strip() for x in f.readlines()]

    rgb_cum_sum = np.zeros((3,))
    cum_num_pixels = 0.
    for cur_img_name in train_img_names:
        cur_img = skimage.io.imread(cur_img_name+'.jpg')
        cur_rgb_sum = np.sum(cur_img, axis=(0,1))
        rgb_cum_sum = rgb_cum_sum + cur_rgb_sum
        cum_num_pixels += np.cumprod(np.shape(cur_img))[1]

    mean_rgb_val = rgb_cum_sum/cum_num_pixels
    print mean_rgb_val
    
elif DATASET=='HRF':
    
    train_set_txt_path = '/home/syshin/Documents/CA/HRF/train_fr.txt'

    with open(train_set_txt_path) as f:
        train_img_names = [x.strip() for x in f.readlines()]

    rgb_cum_sum = np.zeros((3,))
    cum_num_pixels = 0.
    for cur_img_name in train_img_names:
        cur_img = skimage.io.imread(cur_img_name+'.jpg')
        cur_rgb_sum = np.sum(cur_img, axis=(0,1))
        rgb_cum_sum = rgb_cum_sum + cur_rgb_sum
        cum_num_pixels += np.cumprod(np.shape(cur_img))[1]

    mean_rgb_val = rgb_cum_sum/cum_num_pixels
    print mean_rgb_val
    
elif DATASET=='ALL':
    
    dataset_root_path = '/mnt/hdd1/DRIVE'
    train_img_names = sorted(os.listdir(os.path.join(dataset_root_path, 'training/images')))
    train_img_names = map(lambda x: x[:2], train_img_names)
    rgb_cum_sum = np.zeros((3,))
    cum_num_pixels = 0.
    for cur_img_name in train_img_names:
        cur_img = skimage.io.imread(os.path.join(dataset_root_path,'training/images',cur_img_name+'_training.tif'))
        #cur_mask = skimage.io.imread(os.path.join(dataset_root_path,'training/mask',cur_img_name+'_training_mask.gif'))
        #cur_mask = cur_mask>100
        #cur_img = cur_img*np.dstack((cur_mask,cur_mask,cur_mask))
        cur_rgb_sum = np.sum(cur_img, axis=(0,1))
        rgb_cum_sum = rgb_cum_sum + cur_rgb_sum
        #cum_num_pixels += np.sum(cur_mask)
        cum_num_pixels += np.cumprod(cur_img.shape)[1]
        
    dataset_root_path = '/mnt/hdd1/STARE'
    train_img_names = sorted(os.listdir(os.path.join(dataset_root_path, 'stare-images')))
    train_img_names = map(lambda x: x[:6], train_img_names[:10])
    for cur_img_name in train_img_names:
        cur_img = skimage.io.imread(os.path.join(dataset_root_path,'stare-images',cur_img_name+'.ppm'))
        cur_rgb_sum = np.sum(cur_img, axis=(0,1))
        rgb_cum_sum = rgb_cum_sum + cur_rgb_sum
        cum_num_pixels += np.cumprod(np.shape(cur_img))[1]

    mean_rgb_val = rgb_cum_sum/cum_num_pixels    
    print mean_rgb_val