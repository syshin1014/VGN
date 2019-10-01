# to make multiple sub-images for training and test by cropping the whole images
# coded by syshin (180428)

import numpy as np
import os
import skimage.io

IMG_SIZE = [2336,3504]
SUB_IMG_SIZE = 768

if __name__ == '__main__':
    
    save_root_path = '../HRF/all_768'
    if not os.path.isdir(save_root_path):   
        os.mkdir(save_root_path)
    
    set_name = 'train'
    # we apply differnt degrees of overlapping for train & test sets
    # when extract sub-images
    if set_name=='train':
        fr_set_txt_path = '../HRF/train_fr.txt'
        crop_set_txt_path = '../HRF/train_768.txt' 
        
        y_mins = range(0,IMG_SIZE[0]-SUB_IMG_SIZE+1,SUB_IMG_SIZE/2)
        x_mins = range(0,IMG_SIZE[1]-SUB_IMG_SIZE+1,SUB_IMG_SIZE/2)
        y_mins = sorted(list(set(y_mins + [IMG_SIZE[0]-SUB_IMG_SIZE])))
        x_mins = sorted(list(set(x_mins + [IMG_SIZE[1]-SUB_IMG_SIZE])))
        
    elif set_name=='test':
        fr_set_txt_path = '../HRF/test_fr.txt'
        crop_set_txt_path = '../HRF/test_768.txt' 
        
        y_mins = range(0,IMG_SIZE[0]-SUB_IMG_SIZE+1,SUB_IMG_SIZE-50)
        x_mins = range(0,IMG_SIZE[1]-SUB_IMG_SIZE+1,SUB_IMG_SIZE-50)
        y_mins = sorted(list(set(y_mins + [IMG_SIZE[0]-SUB_IMG_SIZE])))
        x_mins = sorted(list(set(x_mins + [IMG_SIZE[1]-SUB_IMG_SIZE])))
    
    with open(fr_set_txt_path) as f:
        img_names = [x.strip() for x in f.readlines()]
        
    file_p = open(crop_set_txt_path, 'w')        

    for cur_img_name in img_names:
        cur_img = skimage.io.imread(cur_img_name+'.jpg')
        cur_mask = skimage.io.imread(cur_img_name+'.tif')
        cur_fov_mask = skimage.io.imread(cur_img_name+'_mask.tif')
        
        for y_idx, cur_y_min in enumerate(y_mins):
            for x_idx, cur_x_min in enumerate(x_mins):
                cur_sub_img = cur_img[cur_y_min:cur_y_min+SUB_IMG_SIZE,cur_x_min:cur_x_min+SUB_IMG_SIZE,:]    
                cur_sub_mask = cur_mask[cur_y_min:cur_y_min+SUB_IMG_SIZE,cur_x_min:cur_x_min+SUB_IMG_SIZE]
                cur_sub_fov_mask = cur_fov_mask[cur_y_min:cur_y_min+SUB_IMG_SIZE,cur_x_min:cur_x_min+SUB_IMG_SIZE,0]
                
                temp = cur_img_name[cur_img_name.rfind('/')+1:]
                
                cur_sub_img_name = os.path.join(save_root_path, temp + '_{:d}_{:d}'.format(y_idx,x_idx))
                
                file_p.write(cur_sub_img_name+'\n')
                file_p.flush()
                
                cur_save_path = os.path.join(cur_sub_img_name + '.bmp')
                skimage.io.imsave(cur_save_path, cur_sub_img)
                cur_save_path = os.path.join(cur_sub_img_name + '.tif')
                skimage.io.imsave(cur_save_path, cur_sub_mask)
                cur_save_path = os.path.join(cur_sub_img_name + '_mask.tif')
                skimage.io.imsave(cur_save_path, cur_sub_fov_mask)
                
    file_p.close()