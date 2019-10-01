# a special script for testing images in the HRF dataset  
# here, multiple sub-images from a single image are independently tested
# and tiled to make a result for the whole image
# coded by syshin (180430)
# updated by syshin (180903)

import numpy as np
import os
import pdb
import argparse
import skimage.io
import networkx as nx
import pickle as pkl
import multiprocessing
import skfmm
import tensorflow as tf

from config import cfg
from model import vessel_segm_vgn
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a vessel_segm_vgn network')
    parser.add_argument('--dataset', default='HRF', help='Dataset to use', type=str)
    #parser.add_argument('--use_multiprocessing', action='store_true', default=False, help='Whether to use the python multiprocessing module')
    parser.add_argument('--use_multiprocessing', default=True, help='Whether to use the python multiprocessing module', type=bool)
    parser.add_argument('--multiprocessing_num_proc', default=8, help='Number of CPU processes to use', type=int)
    parser.add_argument('--win_size', default=32, help='Window size for srns', type=int) # for srns # [4,8,16,32]
    parser.add_argument('--edge_type', default='srns_geo_dist_binary', \
                        help='Graph edge type: Can be srns_geo_dist_binary or srns_geo_dist_weighted', type=str)
    parser.add_argument('--edge_geo_dist_thresh', default=80, help='Threshold for geodesic distance', type=float) # [10,20,40,80]
    parser.add_argument('--model_path', default='../models/HRF/VGN/VGN_HRF.ckpt', \
                        help='Path for a trained model(.ckpt)', type=str)
    parser.add_argument('--save_root', default='../models/HRF/VGN', \
                        help='Root path to save test results', type=str)
    
    ### cnn module related ###    
    parser.add_argument('--cnn_model', default='driu_large', help='CNN model to use', type=str)
    parser.add_argument('--cnn_loss_on', default=True, help='Whether to use a cnn loss for training', type=bool)
    
    ### gnn module related ###
    parser.add_argument('--gnn_loss_on', default=True, help='Whether to use a gnn loss for training', type=bool)
    parser.add_argument('--gnn_loss_weight', default=1., help='Relative weight on the gnn loss', type=float)
    # gat #
    parser.add_argument('--gat_n_heads', default=[4,4], help='Numbers of heads in each layer', type=list) # [4,1]
    #parser.add_argument('--gat_n_heads', nargs='+', help='Numbers of heads in each layer', type=int) # [4,1]
    parser.add_argument('--gat_hid_units', default=[16], help='Numbers of hidden units per each attention head in each layer', type=list)
    #parser.add_argument('--gat_hid_units', nargs='+', help='Numbers of hidden units per each attention head in each layer', type=int)
    parser.add_argument('--gat_use_residual', action='store_true', default=False, help='Whether to use residual learning in GAT')
    
    ### inference module related ###
    parser.add_argument('--norm_type', default=None, help='Norm. type', type=str)
    parser.add_argument('--use_enc_layer', action='store_true', default=False, \
                        help='Whether to use additional conv. layers in a infer_module')
    parser.add_argument('--infer_module_loss_masking_thresh', default=0.05, \
                        help='Threshold for loss masking', type=float)
    parser.add_argument('--infer_module_kernel_size', default=3, \
                        help='Conv. kernel size for the inference module', type=int)
    parser.add_argument('--infer_module_grad_weight', default=1., \
                        help='Relative weight of the grad. on the infer_module', type=float)

    ### training (declared but not used) ###
    parser.add_argument('--do_simul_training', default=True, \
                        help='Whether to train the gnn and inference modules simultaneously or not', type=bool)
    parser.add_argument('--max_iters', default=100000, help='Maximum number of iterations', type=int)
    parser.add_argument('--old_net_ft_lr', default=1e-03, help='Learnining rate for fine-tuning of old parts of network', type=float)
    parser.add_argument('--new_net_lr', default=1e-03, help='Learnining rate for a new part of network', type=float)
    parser.add_argument('--opt', default='adam', help='Optimizer to use: Can be sgd or adam', type=str)
    parser.add_argument('--lr_scheduling', default='pc', help='How to change the learning rate during training', type=str)
    parser.add_argument('--lr_decay_tp', default=1., help='When to decrease the lr during training', type=float) # for pc


    args = parser.parse_args()
    return args


def save_dict(dic, filename):
    with open(filename, 'wb') as f:
        pkl.dump(dic, f)


def load_dict(filename):
    with open(filename, 'rb') as f:
        dic = pkl.load(f)
    return dic


# This was modified to include loading CNN results due to a memory problem 
def make_graph_using_srns((res_file_path, edge_type, win_size, edge_geo_dist_thresh)):

    if 'srns' not in edge_type:
        raise NotImplementedError
    
    # loading    
    cur_res = load_dict(res_file_path)
    fg_prob_map = cur_res['temp_fg_prob_map']
    img_path = cur_res['img_path']
    
    # find local maxima
    vesselness = fg_prob_map
    
    im_y = vesselness.shape[0]
    im_x = vesselness.shape[1]
    y_quan = range(0,im_y,win_size)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0,im_x,win_size)
    x_quan = sorted(list(set(x_quan) | set([im_x])))
    
    max_val = []
    max_pos = []
    for y_idx in xrange(len(y_quan)-1):
        for x_idx in xrange(len(x_quan)-1):
            cur_patch = vesselness[y_quan[y_idx]:y_quan[y_idx+1],x_quan[x_idx]:x_quan[x_idx+1]]
            if np.sum(cur_patch)==0:
                max_val.append(0)
                max_pos.append((y_quan[y_idx]+cur_patch.shape[0]/2,x_quan[x_idx]+cur_patch.shape[1]/2))
            else:
                max_val.append(np.amax(cur_patch))
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[y_idx]+temp[0],x_quan[x_idx]+temp[1]))
    
    graph = nx.Graph()
            
    # add nodes
    for node_idx, (node_y, node_x) in enumerate(max_pos):
        graph.add_node(node_idx, kind='MP', y=node_y, x=node_x, label=node_idx)
        print 'node label', node_idx, 'pos', (node_y,node_x), 'added'

    speed = vesselness

    node_list = list(graph.nodes)
    for i, n in enumerate(node_list): 
            
        phi = np.ones_like(speed)
        phi[graph.node[n]['y'],graph.node[n]['x']] = -1
        if speed[graph.node[n]['y'],graph.node[n]['x']]==0:
            continue

        neighbor = speed[max(0,graph.node[n]['y']-1):min(im_y,graph.node[n]['y']+2), \
                         max(0,graph.node[n]['x']-1):min(im_x,graph.node[n]['x']+2)]
        if np.mean(neighbor)<0.1:
            continue
               
        tt = skfmm.travel_time(phi, speed, narrow=edge_geo_dist_thresh) # travel time

        for n_comp in node_list[i+1:]:
            geo_dist = tt[graph.node[n_comp]['y'],graph.node[n_comp]['x']] # travel time
            if geo_dist < edge_geo_dist_thresh:
                graph.add_edge(n, n_comp, weight=edge_geo_dist_thresh/(edge_geo_dist_thresh+geo_dist))
                print 'An edge BTWN', 'node', n, '&', n_comp, 'is constructed'
    
    # (re-)save
    cur_res['graph'] = graph
    save_dict(cur_res, res_file_path)
    print 'generated a graph for '+img_path


if __name__ == '__main__':
    
    args = parse_args()
    
    print('Called with args:')
    print(args)
    
    # added for testing on a restricted test set
    test_type = ['dr', 'g', 'h']
    # added for testing on a restricted test set
    
    IMG_SIZE = [2336,3504]
    SUB_IMG_SIZE = 768
    SUB_IMG_ROOT_PATH = '../HRF/all_768'
    y_mins = range(0,IMG_SIZE[0]-SUB_IMG_SIZE+1,SUB_IMG_SIZE-50)
    x_mins = range(0,IMG_SIZE[1]-SUB_IMG_SIZE+1,SUB_IMG_SIZE-50)
    y_mins = sorted(list(set(y_mins + [IMG_SIZE[0]-SUB_IMG_SIZE])))
    x_mins = sorted(list(set(x_mins + [IMG_SIZE[1]-SUB_IMG_SIZE])))
    
    with open('../HRF/test_fr.txt') as f:
        test_whole_img_paths = [x.strip() for x in f.readlines()]
        
    # added for testing on a restricted test set
    temp = []
    for i in xrange(len(test_whole_img_paths)):
        for j in xrange(len(test_type)):
            if test_type[j] in test_whole_img_paths[i][util.find(test_whole_img_paths[i],'/')[-1]+1:]:
                temp.append(test_whole_img_paths[i])
                break
    test_whole_img_paths = temp 
    # added for testing on a restricted test set  
        
    test_whole_img_names = map(lambda x: x[util.find(x,'/')[-1]+1:], test_whole_img_paths) # different to 'test_img_names'

    if args.use_multiprocessing:    
        pool = multiprocessing.Pool(processes=args.multiprocessing_num_proc)

    #temp_graph_save_path = args.save_root + '/' + cfg.TRAIN.TEMP_GRAPH_SAVE_PATH if len(args.save_root)>0 else cfg.TRAIN.TEMP_GRAPH_SAVE_PATH   
    res_save_path = args.save_root + '/' + cfg.TEST.RES_SAVE_PATH if len(args.save_root)>0 else cfg.TEST.RES_SAVE_PATH
    
    if len(args.save_root)>0 and not os.path.isdir(args.save_root):
        os.mkdir(args.save_root)
    if not os.path.isdir(res_save_path):   
        os.mkdir(res_save_path)
        
    with open('../HRF/test_768.txt') as f:
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
    
    data_layer_test = util.DataLayer(test_img_names, \
                                     is_training=False, \
                                     use_padding=False)
    
    network = vessel_segm_vgn(args, None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True    
    sess = tf.InteractiveSession(config=config)
    
    saver = tf.train.Saver()  

    sess.run(tf.global_variables_initializer())
    assert args.model_path, 'Model path is not available'  
    print "Loading model..."
    saver.restore(sess, args.model_path)

    f_log = open(os.path.join(res_save_path,'log.txt'), 'w')
    f_log.write(str(args)+'\n')
    f_log.flush()
    timer = util.Timer()
    
    all_labels = np.concatenate(map(lambda x: np.expand_dims(skimage.io.imread(x+'.tif'), axis=0), test_whole_img_paths), axis=0)
    all_masks = np.concatenate(map(lambda x: np.expand_dims(skimage.io.imread(x+'_mask.tif'), axis=0), test_whole_img_paths), axis=0)
    all_masks = all_masks[:,:,:,0]
    all_labels = ((all_labels.astype(float)/255)>=0.5).astype(float)
    all_masks = ((all_masks.astype(float)/255)>=0.5).astype(float)
    all_preds_cum_sum = np.zeros(all_labels.shape)
    all_preds_cum_num = np.zeros(all_labels.shape)
    
    print("Testing the model...")
    
    ### make cnn results (sub-image-wise) ###
    res_file_path_list = []
    for _ in xrange(int(np.ceil(float(len_test)/cfg.TRAIN.BATCH_SIZE))):
        
        # get one batch
        img_list, blobs_test = data_layer_test.forward()
        
        img = blobs_test['img']
        label = blobs_test['label']
        
        conv_feats, fg_prob_tensor, \
        cnn_feat_dict, cnn_feat_spatial_sizes_dict = sess.run(
        [network.conv_feats,
         network.img_fg_prob,
         network.cnn_feat,
         network.cnn_feat_spatial_sizes],
        feed_dict={
            network.imgs: img,
            network.labels: label
            })
    
        cur_batch_size = len(img_list)
        for img_idx in xrange(cur_batch_size):
            cur_res = {}
            cur_res['img_path'] = img_list[img_idx]
            cur_res['img'] = img[[img_idx],:,:,:]
            cur_res['label'] = label[[img_idx],:,:,:]
            cur_res['conv_feats'] = conv_feats[[img_idx],:,:,:]
            cur_res['temp_fg_prob_map'] = fg_prob_tensor[img_idx,:,:,0]
            cur_res['cnn_feat'] = {k: v[[img_idx],:,:,:] for k, v in zip(cnn_feat_dict.keys(), cnn_feat_dict.values())}
            cur_res['cnn_feat_spatial_sizes'] = cnn_feat_spatial_sizes_dict
            cur_res['graph'] = None # will be filled at the next step
            cur_res['final_fg_prob_map'] = cur_res['temp_fg_prob_map']
            cur_res['ap_list'] = []
            
            img_name = img_list[img_idx]
            temp = img_name[util.find(img_name,'/')[-1]:]
            mask = skimage.io.imread(SUB_IMG_ROOT_PATH + temp +'_mask.tif')
            mask = ((mask.astype(float)/255)>=0.5).astype(float)
            cur_res['mask'] = mask
            
            # compute the current AP
            cur_label = label[img_idx,:,:,0]
            label_roi = cur_label[mask.astype(bool)].reshape((-1))
            fg_prob_map_roi = cur_res['temp_fg_prob_map'][mask.astype(bool)].reshape((-1))
            _, cur_cnn_ap = util.get_auc_ap_score(label_roi, fg_prob_map_roi)
            cur_res['ap'] = cur_cnn_ap
            cur_res['ap_list'].append(cur_cnn_ap)

            # (initial) save
            cur_res_file_path = res_save_path + temp + '.pkl'
            save_dict(cur_res, cur_res_file_path)
            res_file_path_list.append(cur_res_file_path)
            
    ### make final results (sub-image-wise) ###         
    # make graphs & append it to the existing pickle files # re-save
    func_arg = []
    for img_idx in xrange(len(res_file_path_list)):
        func_arg.append((res_file_path_list[img_idx], args.edge_type, args.win_size, args.edge_geo_dist_thresh))
    if args.use_multiprocessing:    
            pool.map(make_graph_using_srns, func_arg)
    else:
        for x in func_arg:
            make_graph_using_srns(x)
        
    # make final results 
    for img_idx in xrange(len(res_file_path_list)):
        
        # load
        cur_res = load_dict(res_file_path_list[img_idx])
        
        cur_img = cur_res['img']
        cur_conv_feats = cur_res['conv_feats']
        cur_cnn_feat = cur_res['cnn_feat']
        cur_cnn_feat_spatial_sizes = cur_res['cnn_feat_spatial_sizes']  
        cur_graph = cur_res['graph']
        
        cur_graph = nx.convert_node_labels_to_integers(cur_graph)
        node_byxs = util.get_node_byx_from_graph(cur_graph, [cur_graph.number_of_nodes()])
            
        if 'geo_dist_weighted' in args.edge_type:
            adj = nx.adjacency_matrix(cur_graph)
        else:
            adj = nx.adjacency_matrix(cur_graph,weight=None).astype(float)    

        adj_norm = util.preprocess_graph_gat(adj)
            
        cur_feed_dict = \
        {
        network.imgs: cur_img,
        network.conv_feats: cur_conv_feats,                           
        network.node_byxs: node_byxs,
        network.adj: adj_norm, 
        network.is_lr_flipped: False,
        network.is_ud_flipped: False
        }
        cur_feed_dict.update({network.cnn_feat[cur_key]: cur_cnn_feat[cur_key] for cur_key in network.cnn_feat.keys()})
        cur_feed_dict.update({network.cnn_feat_spatial_sizes[cur_key]: cur_cnn_feat_spatial_sizes[cur_key] for cur_key in network.cnn_feat_spatial_sizes.keys()})
    
        res_prob_map = sess.run(
        [network.post_cnn_img_fg_prob],
        feed_dict=cur_feed_dict)
        
        res_prob_map = res_prob_map[0]
        res_prob_map = res_prob_map.reshape((res_prob_map.shape[1], res_prob_map.shape[2]))
        
        # compute the current AP
        cur_label = cur_res['label']
        cur_label = np.squeeze(cur_label)
        cur_mask = cur_res['mask']
        label_roi = cur_label[cur_mask.astype(bool)].reshape((-1))
        fg_prob_map_roi = res_prob_map[cur_mask.astype(bool)].reshape((-1))
        _, cur_ap = util.get_auc_ap_score(label_roi, fg_prob_map_roi)
        res_prob_map = res_prob_map*cur_mask
            
        cur_res['ap'] = cur_ap
        cur_res['ap_list'].append(cur_ap)
        cur_res['final_fg_prob_map'] = res_prob_map
        cur_res['temp_fg_prob_map'] = res_prob_map    
                
        # (re-)save
        save_dict(cur_res, res_file_path_list[img_idx])
    
    ### aggregate final results ### 
    for cur_res_file_path in res_file_path_list:
        
        # load
        cur_res = load_dict(cur_res_file_path)
        
        res_prob_map = cur_res['final_fg_prob_map']
        
        img_name = cur_res['img_path']
        temp_name = img_name[util.find(img_name,'/')[-1]+1:]
        
        temp_name_splits = temp_name.split('_')
        
        img_idx = test_whole_img_names.index(temp_name_splits[0]+'_'+temp_name_splits[1])
        cur_y_min = y_mins[int(temp_name_splits[2])]
        cur_x_min = x_mins[int(temp_name_splits[3])]
        
        all_preds_cum_sum[img_idx,cur_y_min:cur_y_min+SUB_IMG_SIZE,cur_x_min:cur_x_min+SUB_IMG_SIZE] += res_prob_map
        all_preds_cum_num[img_idx,cur_y_min:cur_y_min+SUB_IMG_SIZE,cur_x_min:cur_x_min+SUB_IMG_SIZE] += 1
        
        print 'AP list for ' + img_name + ' : ' + str(cur_res['ap_list'])
        f_log.write('AP list for ' + img_name + ' : ' + str(cur_res['ap_list']) + '\n')
        
    all_preds = np.divide(all_preds_cum_sum,all_preds_cum_num)
    
    all_labels_roi = all_labels[all_masks.astype(bool)]
    all_preds_roi = all_preds[all_masks.astype(bool)]

    # save qualitative results          
    reshaped_fg_prob_map = all_preds*all_masks
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
        
    all_labels = np.reshape(all_labels, (-1))
    all_preds = np.reshape(all_preds, (-1))
    all_labels_roi = np.reshape(all_labels_roi, (-1))
    all_preds_roi = np.reshape(all_preds_roi, (-1))
        
    auc_test, ap_test = util.get_auc_ap_score(all_labels, all_preds)
    all_labels_bin = np.copy(all_labels).astype(np.bool)
    all_preds_bin = all_preds>=0.5
    all_correct = all_labels_bin==all_preds_bin
    acc_test = np.mean(all_correct.astype(np.float32))
    
    auc_test_roi, ap_test_roi = util.get_auc_ap_score(all_labels_roi, all_preds_roi)
    all_labels_bin_roi = np.copy(all_labels_roi).astype(np.bool)
    all_preds_bin_roi = all_preds_roi>=0.5
    all_correct_roi = all_labels_bin_roi==all_preds_bin_roi
    acc_test_roi = np.mean(all_correct_roi.astype(np.float32))
    
    print 'test_acc: %.4f, test_auc: %.4f, test_ap: %.4f'%(acc_test, auc_test, ap_test)
    print 'test_acc_roi: %.4f, test_auc_roi: %.4f, test_ap_roi: %.4f'%(acc_test_roi, auc_test_roi, ap_test_roi)
    
    f_log.write('test_acc '+str(acc_test)+'\n')
    f_log.write('test_auc '+str(auc_test)+'\n')
    f_log.write('test_ap '+str(ap_test)+'\n')
    f_log.write('test_acc_roi '+str(acc_test_roi)+'\n')
    f_log.write('test_auc_roi '+str(auc_test_roi)+'\n')
    f_log.write('test_ap_roi '+str(ap_test_roi)+'\n')

    f_log.flush()
    
    f_log.close()
    sess.close()
    if args.use_multiprocessing:
        pool.terminate()
    print("Test complete.")