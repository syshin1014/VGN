# a special script for testing images in the HRF dataset  
# here, multiple sub-images from a single image are independently tested
# and tiled to make a result for the whole image
# coded by syshin (180430)

import numpy as np
import os
import pdb
import skimage.io
import argparse
import tensorflow as tf

from config import cfg
from model import vessel_segm_cnn
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a vessel_segm_cnn network')
    parser.add_argument('--dataset', default='HRF', help='Dataset to use', type=str)
    parser.add_argument('--cnn_model', default='driu_large', help='CNN model to use', type=str)
    parser.add_argument('--opt', default='sgd', help='Optimizer to use: Can be sgd or adam', type=str) # declared but not used
    parser.add_argument('--lr', default=1e-03, help='Learning rate to use: Can be any floating point number', type=float) # declared but not used
    parser.add_argument('--lr_decay', default='pc', help='Learning rate decay to use: Can be pc or exp', type=str) # declared but not used
    parser.add_argument('--max_iters', default=100000, help='Maximum number of iterations', type=int) # declared but not used
    parser.add_argument('--model_path', default='../models/HRF/DRIU*/DRIU_HRF.ckpt', help='path for a model(.ckpt) to load', type=str)
    parser.add_argument('--save_root', default='DRIU_HRF', help='root path to save test results', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    
    print('Called with args:')
    print(args)
    
    # added for testing on a restricted test set
    test_type = ['dr', 'g', 'h']
    # added for testing on a restricted test set
     
    with open('../HRF/test_fr.txt') as f:
        test_whole_img_paths = [x.strip() for x in f.readlines()] # different to 'test_img_names'
    
    # added for testing on a restricted test set
    temp = []
    for i in xrange(len(test_whole_img_paths)):
        for j in xrange(len(test_type)):
            if test_type[j] in test_whole_img_paths[i][util.find(test_whole_img_paths[i],'/')[-1]+1:]:
                temp.append(test_whole_img_paths[i])
                break
    test_whole_img_paths = temp 
    # added for testing on a restricted test set
        
    test_whole_img_names = map(lambda x: x[util.find(x,'/')[-1]+1:], test_whole_img_paths)
        
    test_set_txt_path = cfg.TEST.HRF_SET_TXT_PATH
    IMG_SIZE = [2336,3504]
    SUB_IMG_SIZE = 768
    y_mins = range(0,IMG_SIZE[0]-SUB_IMG_SIZE+1,SUB_IMG_SIZE-50)
    x_mins = range(0,IMG_SIZE[1]-SUB_IMG_SIZE+1,SUB_IMG_SIZE-50)
    y_mins = sorted(list(set(y_mins + [IMG_SIZE[0]-SUB_IMG_SIZE])))
    x_mins = sorted(list(set(x_mins + [IMG_SIZE[1]-SUB_IMG_SIZE])))
    
    with open(test_set_txt_path) as f:
        test_img_names = [x.strip() for x in f.readlines()]
     
    # added for testing on a restricted test set
    temp = []
    for i in xrange(len(test_img_names)):
        for j in xrange(len(test_type)):
            if test_type[j] in test_img_names[i][util.find(test_img_names[i],'/')[-1]+1:]:
                temp.append(test_img_names[i])
                break
    test_img_names = temp
    # added for testing on a restricted test set

    len_test = len(test_img_names)
    
    data_layer_test = util.DataLayer(test_img_names, is_training=False)
    
    res_save_path = args.save_root + '/' + cfg.TEST.WHOLE_IMG_RES_SAVE_PATH if len(args.save_root)>0 else cfg.TEST.WHOLE_IMG_RES_SAVE_PATH
    if len(args.save_root)>0 and not os.path.isdir(args.save_root):  
        os.mkdir(args.save_root)
    if not os.path.isdir(res_save_path):   
        os.mkdir(res_save_path) 
    
    network = vessel_segm_cnn(args, None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    sess = tf.InteractiveSession(config=config)
    saver = tf.train.Saver()  

    sess.run(tf.global_variables_initializer())
    
    assert args.model_path, 'Model path is not available'  
    print "Loading model..."
    saver.restore(sess, args.model_path)

    f_log = open(os.path.join(res_save_path,'log.txt'), 'w')
    f_log.write(args.model_path+'\n')
    timer = util.Timer()
     
    all_cnn_labels = np.concatenate(map(lambda x: np.expand_dims(skimage.io.imread(x+'.tif'), axis=0), test_whole_img_paths), axis=0)
    all_cnn_masks = np.concatenate(map(lambda x: np.expand_dims(skimage.io.imread(x+'_mask.tif'), axis=0), test_whole_img_paths), axis=0)
    all_cnn_masks = all_cnn_masks[:,:,:,0]
    all_cnn_labels = ((all_cnn_labels.astype(float)/255)>=0.5).astype(float)
    all_cnn_masks = ((all_cnn_masks.astype(float)/255)>=0.5).astype(float)
    all_cnn_preds_cum_sum = np.zeros(all_cnn_labels.shape)
    all_cnn_preds_cum_num = np.zeros(all_cnn_labels.shape)

    for _ in xrange(int(np.ceil(float(len_test)/cfg.TRAIN.BATCH_SIZE))):
                
        timer.tic()
        
        # get one batch
        img_list, blobs_test = data_layer_test.forward()
        
        img = blobs_test['img']
        label = blobs_test['label']

        fg_prob_map = sess.run(
        [network.fg_prob],
        feed_dict={
            network.is_training: False,
            network.imgs: img,
            network.labels: label,
            })
        fg_prob_map = fg_prob_map[0]
        fg_prob_map = fg_prob_map.reshape((fg_prob_map.shape[1],fg_prob_map.shape[2]))
    
        timer.toc()
        
        cur_batch_size = len(img_list)
        for i in xrange(cur_batch_size):
            cur_test_img_path = img_list[i]
            temp_name = cur_test_img_path[util.find(cur_test_img_path,'/')[-1]+1:]
            
            temp_name_splits = temp_name.split('_')
            
            img_idx = test_whole_img_names.index(temp_name_splits[0]+'_'+temp_name_splits[1])
            cur_y_min = y_mins[int(temp_name_splits[2])]
            cur_x_min = x_mins[int(temp_name_splits[3])]
            
            all_cnn_preds_cum_sum[img_idx,cur_y_min:cur_y_min+SUB_IMG_SIZE,cur_x_min:cur_x_min+SUB_IMG_SIZE] += fg_prob_map
            all_cnn_preds_cum_num[img_idx,cur_y_min:cur_y_min+SUB_IMG_SIZE,cur_x_min:cur_x_min+SUB_IMG_SIZE] += 1
        
    all_cnn_preds = np.divide(all_cnn_preds_cum_sum,all_cnn_preds_cum_num)
    
    all_cnn_labels_roi = all_cnn_labels[all_cnn_masks.astype(bool)]
    all_cnn_preds_roi = all_cnn_preds[all_cnn_masks.astype(bool)]

    # save qualitative results          
    reshaped_fg_prob_map = all_cnn_preds*all_cnn_masks
    reshaped_output = reshaped_fg_prob_map>=0.5
    for img_idx, temp_name in enumerate(test_whole_img_names):
        
        cur_reshaped_fg_prob_map = (reshaped_fg_prob_map[img_idx,:,:]*255).astype(int)
        cur_reshaped_fg_prob_map_inv = ((1.-reshaped_fg_prob_map[img_idx,:,:])*255).astype(int)
        cur_reshaped_output = reshaped_output[img_idx,:,:].astype(int)*255

        cur_fg_prob_save_path = os.path.join(res_save_path, temp_name + '_prob.png')
        cur_fg_prob_inv_save_path = os.path.join(res_save_path, temp_name + '_prob_inv.png')
        cur_output_save_path = os.path.join(res_save_path, temp_name + '_output.png')
        cur_numpy_save_path = os.path.join(res_save_path, temp_name + '.npy')
        
        skimage.io.imsave(cur_fg_prob_save_path, cur_reshaped_fg_prob_map)
        skimage.io.imsave(cur_fg_prob_inv_save_path, cur_reshaped_fg_prob_map_inv)
        skimage.io.imsave(cur_output_save_path, cur_reshaped_output)
        np.save(cur_numpy_save_path, reshaped_fg_prob_map[img_idx,:,:])
        
    all_cnn_labels = np.reshape(all_cnn_labels, (-1))
    all_cnn_preds = np.reshape(all_cnn_preds, (-1))
    all_cnn_labels_roi = np.reshape(all_cnn_labels_roi, (-1))
    all_cnn_preds_roi = np.reshape(all_cnn_preds_roi, (-1))
        
    cnn_auc_test, cnn_ap_test = util.get_auc_ap_score(all_cnn_labels, all_cnn_preds)
    all_cnn_labels_bin = np.copy(all_cnn_labels).astype(np.bool)
    all_cnn_preds_bin = all_cnn_preds>=0.5
    all_cnn_correct = all_cnn_labels_bin==all_cnn_preds_bin
    cnn_acc_test = np.mean(all_cnn_correct.astype(np.float32))
    
    cnn_auc_test_roi, cnn_ap_test_roi = util.get_auc_ap_score(all_cnn_labels_roi, all_cnn_preds_roi)
    all_cnn_labels_bin_roi = np.copy(all_cnn_labels_roi).astype(np.bool)
    all_cnn_preds_bin_roi = all_cnn_preds_roi>=0.5
    all_cnn_correct_roi = all_cnn_labels_bin_roi==all_cnn_preds_bin_roi
    cnn_acc_test_roi = np.mean(all_cnn_correct_roi.astype(np.float32))
    
    print 'test_cnn_acc: %.4f, test_cnn_auc: %.4f, test_cnn_ap: %.4f'%(cnn_acc_test, cnn_auc_test, cnn_ap_test)
    print 'test_cnn_acc_roi: %.4f, test_cnn_auc_roi: %.4f, test_cnn_ap_roi: %.4f'%(cnn_acc_test_roi, cnn_auc_test_roi, cnn_ap_test_roi)
    
    f_log.write('test_cnn_acc '+str(cnn_acc_test)+'\n')
    f_log.write('test_cnn_auc '+str(cnn_auc_test)+'\n')
    f_log.write('test_cnn_ap '+str(cnn_ap_test)+'\n')
    f_log.write('test_cnn_acc_roi '+str(cnn_acc_test_roi)+'\n')
    f_log.write('test_cnn_auc_roi '+str(cnn_auc_test_roi)+'\n')
    f_log.write('test_cnn_ap_roi '+str(cnn_ap_test_roi)+'\n')

    f_log.flush()
    
    print 'speed: {:.3f}s'.format(timer.average_time)
    
    f_log.close()
    sess.close()
    print("Test complete.")