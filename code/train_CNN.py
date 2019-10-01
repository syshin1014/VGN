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
    parser = argparse.ArgumentParser(description='Train a vessel_segm_cnn network')
    parser.add_argument('--dataset', default='DRIVE', help='Dataset to use: Can be DRIVE or STARE or CHASE_DB1 or HRF', type=str)
    parser.add_argument('--cnn_model', default='driu', help='CNN model to use', type=str)
    parser.add_argument('--use_fov_mask', default=True, help='Whether to use fov masks', type=bool)
    parser.add_argument('--opt', default='adam', help='Optimizer to use: Can be sgd or adam', type=str)
    parser.add_argument('--lr', default=1e-02, help='Learning rate to use: Can be any floating point number', type=float)
    parser.add_argument('--lr_decay', default='pc', help='Learning rate decay to use: Can be const or pc or exp', type=str)
    parser.add_argument('--max_iters', default=50000, help='Maximum number of iterations', type=int)
    parser.add_argument('--pretrained_model', default='../pretrained_model/VGG_imagenet.npy', help='path for a pretrained model(.npy)', type=str)
    #parser.add_argument('--pretrained_model', default=None, help='path for a pretrained model(.ckpt)', type=str)
    parser.add_argument('--save_root', default='DRIU_DRIVE', help='root path to save trained models and test results', type=str)

    args = parser.parse_args()
    return args


def load(data_path, session, ignore_missing=False):
    data_dict = np.load(data_path).item()
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey in data_dict[key]:
                if subkey=='weights':
                    target_subkey='W'
                elif subkey=='biases':
                    target_subkey='b'
                try:
                    var = tf.get_variable(target_subkey)
                    session.run(var.assign(data_dict[key][subkey]))
                    print "assign pretrain model "+subkey+ " to "+key
                except ValueError:
                    print "ignore "+key+"/"+subkey
                    #print "ignore "+key
                    if not ignore_missing:
                        raise


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
    elif args.dataset=='HRF':
        train_set_txt_path = cfg.TRAIN.HRF_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.HRF_SET_TXT_PATH

    with open(train_set_txt_path) as f:
        train_img_names = [x.strip() for x in f.readlines()]
    with open(test_set_txt_path) as f:
        test_img_names = [x.strip() for x in f.readlines()]
        
    if args.dataset=='HRF':
        test_img_names = map(lambda x: test_img_names[x], range(7,len(test_img_names),20))
        
    len_train = len(train_img_names)
    len_test = len(test_img_names)
    
    data_layer_train = util.DataLayer(train_img_names, is_training=True)
    data_layer_test = util.DataLayer(test_img_names, is_training=False) 
    
    model_save_path = args.save_root + '/' + cfg.TRAIN.MODEL_SAVE_PATH if len(args.save_root)>0 else cfg.TRAIN.MODEL_SAVE_PATH
    res_save_path = args.save_root + '/' + cfg.TEST.RES_SAVE_PATH if len(args.save_root)>0 else cfg.TEST.RES_SAVE_PATH
    if len(args.save_root)>0 and not os.path.isdir(args.save_root):  
        os.mkdir(args.save_root)
    if not os.path.isdir(model_save_path):    
        os.mkdir(model_save_path)
    if not os.path.isdir(res_save_path):   
        os.mkdir(res_save_path)
    
    network = vessel_segm_cnn(args, None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    sess = tf.InteractiveSession(config=config)
    
    saver = tf.train.Saver(max_to_keep=100)
    summary_writer = tf.summary.FileWriter(model_save_path, sess.graph)
    
    sess.run(tf.global_variables_initializer())
    if args.pretrained_model is not None:
        print "Loading model..."
        load(args.pretrained_model, sess, ignore_missing=True)
    
    f_log = open(os.path.join(model_save_path,'log.txt'), 'w')
    last_snapshot_iter = -1
    timer = util.Timer()
    
    train_loss_list = []
    test_loss_list = []
    print("Training the model...")
    for iter in xrange(args.max_iters):
    
        timer.tic()
        
        # get one batch
        _, blobs_train = data_layer_train.forward()  

        if args.use_fov_mask:
            fov_masks = blobs_train['fov']
        else:
            fov_masks = np.ones(blobs_train['label'].shape, dtype=blobs_train['label'].dtype)      
        
        _, loss_val, accuracy_val, pre_val, rec_val = sess.run(
                [network.train_op, network.loss, network.accuracy, network.precision, network.recall],
                feed_dict={
                    network.is_training: True,
                    network.imgs: blobs_train['img'],
                    network.labels: blobs_train['label'],
                    network.fov_masks: fov_masks
                    })
       
        timer.toc()
        train_loss_list.append(loss_val)
    
        if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
            print 'iter: %d / %d, loss: %.4f, accuracy: %.4f, precision: %.4f, recall: %.4f'\
                    %(iter+1, args.max_iters, loss_val, accuracy_val, pre_val, rec_val)     
            print 'speed: {:.3f}s / iter'.format(timer.average_time)
    
        if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
            last_snapshot_iter = iter
            filename = os.path.join(model_save_path,('iter_{:d}'.format(iter+1) + '.ckpt'))
            saver.save(sess, filename)
            print 'Wrote snapshot to: {:s}'.format(filename)
    
        if (iter+1) % cfg.TRAIN.TEST_ITERS == 0:
            
            all_labels = np.zeros((0,))
            all_preds = np.zeros((0,))

            for _ in xrange(int(np.ceil(float(len_test)/cfg.TRAIN.BATCH_SIZE))):
                
                # get one batch
                img_list, blobs_test = data_layer_test.forward()
                
                imgs = blobs_test['img']
                labels = blobs_test['label']
                if args.use_fov_mask:
                    fov_masks = blobs_test['fov']
                else:
                    fov_masks = np.ones(labels.shape, dtype=labels.dtype)
                
                loss_val, fg_prob_map = sess.run(
                [network.loss, network.fg_prob],
                feed_dict={
                    network.is_training: False,
                    network.imgs: imgs,
                    network.labels: labels,
                    network.fov_masks: fov_masks
                    })

                test_loss_list.append(loss_val)
                
                all_labels = np.concatenate((all_labels,np.reshape(labels, (-1))))
                fg_prob_map = fg_prob_map*fov_masks.astype(float)
                all_preds = np.concatenate((all_preds,np.reshape(fg_prob_map, (-1))))
                    
                # save qualitative results
                cur_batch_size = len(img_list)
                reshaped_fg_prob_map = fg_prob_map.reshape((cur_batch_size,fg_prob_map.shape[1],fg_prob_map.shape[2]))
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
            
            auc_test, ap_test = util.get_auc_ap_score(all_labels, all_preds)
            all_labels_bin = np.copy(all_labels).astype(np.bool)
            all_preds_bin = all_preds>=0.5
            all_correct = all_labels_bin==all_preds_bin
            acc_test = np.mean(all_correct.astype(np.float32))
        
            summary = tf.Summary()
            summary.value.add(tag="train_loss", simple_value=float(np.mean(train_loss_list)))
            summary.value.add(tag="test_loss", simple_value=float(np.mean(test_loss_list)))
            summary.value.add(tag="test_acc", simple_value=float(acc_test))
            summary.value.add(tag="test_auc", simple_value=float(auc_test))
            summary.value.add(tag="test_ap", simple_value=float(ap_test))
            summary_writer.add_summary(summary, global_step=iter+1)
            summary_writer.flush()
            
            print 'iter: %d / %d, train_loss: %.4f'%(iter+1, args.max_iters, np.mean(train_loss_list))
            print 'iter: %d / %d, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, test_ap: %.4f'\
                    %(iter+1, args.max_iters, np.mean(test_loss_list), acc_test, auc_test, ap_test)
            
            f_log.write('iter: '+str(iter+1)+' / '+str(args.max_iters)+'\n')
            f_log.write('train_loss '+str(np.mean(train_loss_list))+'\n')
            f_log.write('iter: '+str(iter+1)+' / '+str(args.max_iters)+'\n')
            f_log.write('test_loss '+str(np.mean(test_loss_list))+'\n')
            f_log.write('test_acc '+str(acc_test)+'\n')
            f_log.write('test_auc '+str(auc_test)+'\n')
            f_log.write('test_ap '+str(ap_test)+'\n')
            f_log.flush()
            
            train_loss_list = []
            test_loss_list = []    
    
    if last_snapshot_iter != iter:
        filename = os.path.join(model_save_path,('iter_{:d}'.format(iter+1) + '.ckpt'))
        saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)
    
    f_log.close()
    sess.close()
    print("Training complete.")