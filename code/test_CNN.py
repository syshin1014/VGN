# coded by syshin

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
    parser.add_argument('--dataset', default='DRIVE', help='Dataset to use: Can be DRIVE or STARE or CHASE_DB1', type=str)
    parser.add_argument('--cnn_model', default='driu', help='CNN model to use', type=str)
    parser.add_argument('--use_fov_mask', default=True, help='Whether to use fov masks', type=bool)
    parser.add_argument('--opt', default='adam', help='Optimizer to use: Can be sgd or adam', type=str) # declared but not used
    parser.add_argument('--lr', default=1e-02, help='Learning rate to use: Can be any floating point number', type=float) # declared but not used
    parser.add_argument('--lr_decay', default='pc', help='Learning rate decay to use: Can be pc or exp', type=str) # declared but not used
    parser.add_argument('--max_iters', default=50000, help='Maximum number of iterations', type=int) # declared but not used
    parser.add_argument('--model_path', default='../models/DRIVE/DRIU*/DRIU_DRIVE.ckpt', help='path for a model(.ckpt) to load', type=str)
    parser.add_argument('--save_root', default='DRIU_DRIVE', help='root path to save test results', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    
    print('Called with args:')
    print(args)
    
    if args.dataset=='DRIVE':
        train_set_txt_path = cfg.TRAIN.DRIVE_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.DRIVE_SET_TXT_PATH
    elif args.dataset=='STARE':
        train_set_txt_path = cfg.TRAIN.STARE_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.STARE_SET_TXT_PATH
    elif args.dataset=='CHASE_DB1':
        train_set_txt_path = cfg.TRAIN.CHASE_DB1_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.CHASE_DB1_SET_TXT_PATH
    
    with open(train_set_txt_path) as f:
        train_img_names = [x.strip() for x in f.readlines()]
    with open(test_set_txt_path) as f:
        test_img_names = [x.strip() for x in f.readlines()]
    
    len_train = len(train_img_names) 
    len_test = len(test_img_names)
    
    data_layer_train = util.DataLayer(train_img_names, is_training=False)
    data_layer_test = util.DataLayer(test_img_names, is_training=False)
    
    res_save_path = args.save_root + '/' + cfg.TEST.RES_SAVE_PATH if len(args.save_root)>0 else cfg.TEST.RES_SAVE_PATH
    
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
    
    train_loss_list = []
    for _ in xrange(int(np.ceil(float(len_train)/cfg.TRAIN.BATCH_SIZE))):
                
        timer.tic()
        
        # get one batch
        img_list, blobs_train = data_layer_train.forward()
        
        img = blobs_train['img']
        label = blobs_train['label']
        if args.use_fov_mask:
            fov_mask = blobs_train['fov']
        else:
            fov_mask = np.ones(label.shape, dtype=label.dtype)  
        
        loss_val, fg_prob_map = sess.run(
        [network.loss, network.fg_prob],
        feed_dict={
            network.is_training: False,
            network.imgs: img,
            network.labels: label,
            network.fov_masks: fov_mask
            })
    
        timer.toc()
        
        #fg_prob_map = fg_prob_map*fov_mask.astype(float)

        train_loss_list.append(loss_val)
            
        # save qualitative results
        cur_batch_size = len(img_list)
        reshaped_fg_prob_map = fg_prob_map.reshape((cur_batch_size,fg_prob_map.shape[1],fg_prob_map.shape[2]))
        if args.dataset=='DRIVE':
            if args.dataset=='DRIVE':
                mask = np.concatenate(map(lambda x: np.expand_dims(skimage.io.imread(x+'_mask.gif'), axis=0), img_list), axis=0)
            else:
                mask = np.concatenate(map(lambda x: np.expand_dims(skimage.io.imread(x+'_mask.tif'), axis=0), img_list), axis=0)
            mask = ((mask.astype(float)/255)>=0.5).astype(float)
            reshaped_fg_prob_map = reshaped_fg_prob_map*mask
        reshaped_output = reshaped_fg_prob_map>=0.5    
        for img_idx in xrange(cur_batch_size):
            cur_test_img_path = img_list[img_idx]
            temp_name = cur_test_img_path[util.find(cur_test_img_path,'/')[-1]+1:]

            cur_reshaped_fg_prob_map = (reshaped_fg_prob_map[img_idx,:,:]*255).astype(int)
            cur_reshaped_output = reshaped_output[img_idx,:,:].astype(int)*255

            cur_fg_prob_save_path = os.path.join(res_save_path, temp_name + '_prob.png')
            cur_output_save_path = os.path.join(res_save_path, temp_name + '_output.png')
            
            skimage.io.imsave(cur_fg_prob_save_path, cur_reshaped_fg_prob_map)
            skimage.io.imsave(cur_output_save_path, cur_reshaped_output)
    
    test_loss_list = []
    all_cnn_labels = np.zeros((0,))
    all_cnn_preds = np.zeros((0,))
    all_cnn_labels_roi = np.zeros((0,))
    all_cnn_preds_roi = np.zeros((0,))
    for _ in xrange(int(np.ceil(float(len_test)/cfg.TRAIN.BATCH_SIZE))):
                
        timer.tic()
        
        # get one batch
        img_list, blobs_test = data_layer_test.forward()
        
        img = blobs_test['img']
        label = blobs_test['label']     
        if args.use_fov_mask:
            fov_mask = blobs_test['fov']
        else:
            fov_mask = np.ones(label.shape, dtype=label.dtype)  

        loss_val, fg_prob_map = sess.run(
        [network.loss, network.fg_prob],
        feed_dict={
            network.is_training: False,
            network.imgs: img,
            network.labels: label,
            network.fov_masks: fov_mask
            })
    
        timer.toc()
        
        #fg_prob_map = fg_prob_map*fov_mask.astype(float)

        test_loss_list.append(loss_val)

        all_cnn_labels = np.concatenate((all_cnn_labels,np.reshape(label, (-1))))
        all_cnn_preds = np.concatenate((all_cnn_preds,np.reshape(fg_prob_map, (-1))))
        
        # save qualitative results
        cur_batch_size = len(img_list)
        reshaped_fg_prob_map = fg_prob_map.reshape((cur_batch_size,fg_prob_map.shape[1],fg_prob_map.shape[2]))
        
        if args.dataset=='DRIVE':
            if args.dataset=='DRIVE':
                mask = np.concatenate(map(lambda x: np.expand_dims(skimage.io.imread(x+'_mask.gif'), axis=0), img_list), axis=0)
            else:
                mask = np.concatenate(map(lambda x: np.expand_dims(skimage.io.imread(x+'_mask.tif'), axis=0), img_list), axis=0)

            mask = ((mask.astype(float)/255)>=0.5).astype(float)
            label_roi = label[mask.astype(bool)]
            fg_prob_map_roi = fg_prob_map[mask.astype(bool)]
            all_cnn_labels_roi = np.concatenate((all_cnn_labels_roi,np.reshape(label_roi, (-1))))
            all_cnn_preds_roi = np.concatenate((all_cnn_preds_roi,np.reshape(fg_prob_map_roi, (-1))))
            reshaped_fg_prob_map = reshaped_fg_prob_map*mask
            label = np.squeeze(label.astype(float), axis=-1)*mask
        else:
            label = np.squeeze(label.astype(float), axis=-1)

        reshaped_output = reshaped_fg_prob_map>=0.5
        for img_idx in xrange(cur_batch_size):
            cur_test_img_path = img_list[img_idx]
            temp_name = cur_test_img_path[util.find(cur_test_img_path,'/')[-1]+1:]
            
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
                    
    cnn_auc_test, cnn_ap_test = util.get_auc_ap_score(all_cnn_labels, all_cnn_preds)
    all_cnn_labels_bin = np.copy(all_cnn_labels).astype(np.bool)
    all_cnn_preds_bin = all_cnn_preds>=0.5
    all_cnn_correct = all_cnn_labels_bin==all_cnn_preds_bin
    cnn_acc_test = np.mean(all_cnn_correct.astype(np.float32))
    
    if args.dataset=='DRIVE':
        cnn_auc_test_roi, cnn_ap_test_roi = util.get_auc_ap_score(all_cnn_labels_roi, all_cnn_preds_roi)
        all_cnn_labels_bin_roi = np.copy(all_cnn_labels_roi).astype(np.bool)
        all_cnn_preds_bin_roi = all_cnn_preds_roi>=0.5
        all_cnn_correct_roi = all_cnn_labels_bin_roi==all_cnn_preds_bin_roi
        cnn_acc_test_roi = np.mean(all_cnn_correct_roi.astype(np.float32))
    
    #print 'train_loss: %.4f'%(np.mean(train_loss_list))
    print 'test_loss: %.4f'%(np.mean(test_loss_list))
    print 'test_cnn_acc: %.4f, test_cnn_auc: %.4f, test_cnn_ap: %.4f'%(cnn_acc_test, cnn_auc_test, cnn_ap_test)
    if args.dataset=='DRIVE':
        print 'test_cnn_acc_roi: %.4f, test_cnn_auc_roi: %.4f, test_cnn_ap_roi: %.4f'%(cnn_acc_test_roi, cnn_auc_test_roi, cnn_ap_test_roi)
    
    #f_log.write('train_loss '+str(np.mean(train_loss_list))+'\n')
    f_log.write('test_loss '+str(np.mean(test_loss_list))+'\n')
    f_log.write('test_cnn_acc '+str(cnn_acc_test)+'\n')
    f_log.write('test_cnn_auc '+str(cnn_auc_test)+'\n')
    f_log.write('test_cnn_ap '+str(cnn_ap_test)+'\n')
    if args.dataset=='DRIVE':
        f_log.write('test_cnn_acc_roi '+str(cnn_acc_test_roi)+'\n')
        f_log.write('test_cnn_auc_roi '+str(cnn_auc_test_roi)+'\n')
        f_log.write('test_cnn_ap_roi '+str(cnn_ap_test_roi)+'\n')

    f_log.flush()
    
    print 'speed: {:.3f}s'.format(timer.average_time)
    
    f_log.close()
    sess.close()
    print("Test complete.")