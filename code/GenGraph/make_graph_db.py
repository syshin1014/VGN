import numpy as np
import skimage.io
import os
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
from scipy import ndimage
import mahotas as mh
import multiprocessing
import matplotlib.pyplot as plt
import argparse
import skfmm
from scipy.ndimage.morphology import distance_transform_edt

import _init_paths
from bwmorph import bwmorph
from config import cfg
import util

DEBUG = False


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Make a graph db')
    parser.add_argument('--dataset', default='DRIVE', \
                        help='Dataset to use: Can be DRIVE or STARE or CHASE_DB1 or HRF', type=str)
    """parser.add_argument('--use_multiprocessing', action='store_true', \
                        default=False, help='Whether to use the python multiprocessing module')"""
    parser.add_argument('--use_multiprocessing', default=True, \
                        help='Whether to use the python multiprocessing module', type=bool)
    parser.add_argument('--source_type', default='result', \
                        help='Source to be used: Can be result or gt', type=str)
    parser.add_argument('--win_size', default=4, \
                        help='Window size for srns', type=int) # for srns # [4,8,16]
    parser.add_argument('--edge_method', default='geo_dist', \
                        help='Edge construction method: Can be geo_dist or eu_dist', type=str)
    parser.add_argument('--edge_dist_thresh', default=10, \
                        help='Distance threshold for edge construction', type=float) # [10,20,40]

    args = parser.parse_args()
    return args   


def generate_graph_using_srns((img_name, im_root_path, cnn_result_root_path, params)):
         
    win_size_str = '%.2d_%.2d'%(params.win_size,params.edge_dist_thresh)
    
    if params.source_type=='gt':
        win_size_str = win_size_str + '_gt'
    
    if 'DRIVE' in img_name:        
        im_ext = '_image.tif'
        label_ext = '_label.gif'
        len_y = 592
        len_x = 592
    elif 'STARE' in img_name:
        im_ext = '.ppm'
        label_ext = '.ah.ppm'
        len_y = 704
        len_x = 704
    elif 'CHASE_DB1' in img_name:
        im_ext = '.jpg'
        label_ext = '_1stHO.png'
        len_y = 1024
        len_x = 1024
    elif 'HRF' in img_name:
        im_ext = '.bmp'
        label_ext = '.tif'
        len_y = 768
        len_x = 768
    
    cur_filename = img_name[util.find(img_name,'/')[-1]+1:]
    print 'processing '+cur_filename
    cur_im_path = os.path.join(im_root_path, cur_filename+im_ext)
    cur_gt_mask_path = os.path.join(im_root_path, cur_filename+label_ext)
    if params.source_type=='gt': 
        cur_res_prob_path = cur_gt_mask_path
    else:
        cur_res_prob_path = os.path.join(cnn_result_root_path, cur_filename+'_prob.png')
    
    cur_vis_res_im_savepath = os.path.join(cnn_result_root_path, cur_filename+'_'+win_size_str+'_vis_graph_res_on_im.png')
    cur_vis_res_mask_savepath = os.path.join(cnn_result_root_path, cur_filename+'_'+win_size_str+'_vis_graph_res_on_mask.png')
    cur_res_graph_savepath = os.path.join(cnn_result_root_path, cur_filename+'_'+win_size_str+'.graph_res')
    # Note that there is no difference on above paths according to 'params.edge_method'
    
    im = skimage.io.imread(cur_im_path)

    gt_mask = skimage.io.imread(cur_gt_mask_path)
    gt_mask = gt_mask.astype(float)/255
    gt_mask = gt_mask>=0.5
    
    vesselness = skimage.io.imread(cur_res_prob_path)
    vesselness = vesselness.astype(float)/255
    
    temp = np.copy(im)
    im = np.zeros((len_y,len_x,3), dtype=temp.dtype)
    im[:temp.shape[0],:temp.shape[1],:] = temp
    temp = np.copy(gt_mask)
    gt_mask = np.zeros((len_y,len_x), dtype=temp.dtype)
    gt_mask[:temp.shape[0], :temp.shape[1]] = temp
    temp = np.copy(vesselness)
    vesselness = np.zeros((len_y,len_x), dtype=temp.dtype)
    vesselness[:temp.shape[0], :temp.shape[1]] = temp
    
    # find local maxima
    im_y = im.shape[0]
    im_x = im.shape[1]
    y_quan = range(0,im_y,args.win_size)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0,im_x,args.win_size)
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
    if params.source_type=='gt':
        speed = bwmorph(speed, 'dilate', n_iter=1)
        speed = speed.astype(float)
        
    edge_dist_thresh_sq = params.edge_dist_thresh**2

    node_list = list(graph.nodes)
    for i, n in enumerate(node_list):
            
        if speed[graph.node[n]['y'],graph.node[n]['x']]==0:
            continue
        neighbor = speed[max(0,graph.node[n]['y']-1):min(im_y,graph.node[n]['y']+2), \
                         max(0,graph.node[n]['x']-1):min(im_x,graph.node[n]['x']+2)]
        
        if np.mean(neighbor)<0.1:
            continue
        
        if params.edge_method=='geo_dist':
        
            phi = np.ones_like(speed)
            phi[graph.node[n]['y'],graph.node[n]['x']] = -1
            tt = skfmm.travel_time(phi, speed, narrow=params.edge_dist_thresh) # travel time
    
            if DEBUG:
                plt.figure()
                plt.imshow(tt, interpolation='nearest')
                plt.show()
    
                plt.cla()
                plt.clf()
                plt.close()
    
            for n_comp in node_list[i+1:]:
                geo_dist = tt[graph.node[n_comp]['y'],graph.node[n_comp]['x']] # travel time
                if geo_dist < params.edge_dist_thresh:
                    graph.add_edge(n, n_comp, weight=params.edge_dist_thresh/(params.edge_dist_thresh+geo_dist))
                    print 'An edge BTWN', 'node', n, '&', n_comp, 'is constructed'
                    
        elif params.edge_method=='eu_dist':
                
            for n_comp in node_list[i+1:]:
                eu_dist = (graph.node[n_comp]['y']-graph.node[n]['y'])**2 + (graph.node[n_comp]['x']-graph.node[n]['x'])**2
                if eu_dist < edge_dist_thresh_sq:
                    graph.add_edge(n, n_comp, weight=1.)
                    print 'An edge BTWN', 'node', n, '&', n_comp, 'is constructed'
                    
        else:
            raise NotImplementedError
 
    # visualize the constructed graph
    util.visualize_graph(im, graph, show_graph=False, \
                    save_graph=True, num_nodes_each_type=[0,graph.number_of_nodes()], save_path=cur_vis_res_im_savepath)
    util.visualize_graph(gt_mask, graph, show_graph=False, \
                    save_graph=True, num_nodes_each_type=[0,graph.number_of_nodes()], save_path=cur_vis_res_mask_savepath)
    
    # save as files
    nx.write_gpickle(graph, cur_res_graph_savepath, protocol=pkl.HIGHEST_PROTOCOL)
        
    graph.clear()
            
    
if __name__ == '__main__':

    args = parse_args()
    
    print('Called with args:')
    print(args)
    
    if args.dataset=='DRIVE':
        train_set_txt_path = '../../DRIVE/train.txt'
        test_set_txt_path = '../../DRIVE/test.txt'
        im_root_path = '../../DRIVE/all'
        cnn_result_root_path = '../new_exp/DRIVE_cnn/test'
    elif args.dataset=='STARE':
        train_set_txt_path = '../../STARE/train.txt'
        test_set_txt_path = '../../STARE/test.txt'
        im_root_path = '../../STARE/all'
        cnn_result_root_path = '../STARE_cnn/res_resized'
    elif args.dataset=='CHASE_DB1':
        train_set_txt_path = '../../CHASE_DB1/train.txt'
        test_set_txt_path = '../../CHASE_DB1/test.txt'
        im_root_path = '../../CHASE_DB1/all'
        cnn_result_root_path = '../CHASE_cnn/test_resized_graph_gen'
    elif args.dataset=='HRF':
        train_set_txt_path = '../../HRF/train_768.txt'
        test_set_txt_path = '../../HRF/test_768.txt'
        im_root_path = '../../HRF/all_768'
        cnn_result_root_path = '../HRF_cnn/test' 
    
    with open(train_set_txt_path) as f:
        train_img_names = [x.strip() for x in f.readlines()]
    with open(test_set_txt_path) as f:
        test_img_names = [x.strip() for x in f.readlines()]
    
    len_train = len(train_img_names)
    len_test = len(test_img_names)
        
    func = generate_graph_using_srns
    func_arg_train = map(lambda x: (train_img_names[x], im_root_path, cnn_result_root_path, args), xrange(len_train))
    func_arg_test = map(lambda x: (test_img_names[x], im_root_path, cnn_result_root_path, args), xrange(len_test))

    if args.use_multiprocessing:
        pool = multiprocessing.Pool(processes=20)

        pool.map(func, func_arg_train)
        pool.map(func, func_arg_test)

        pool.terminate()
    else:
        for x in func_arg_train:
            func(x)
        for x in func_arg_test:
            func(x)