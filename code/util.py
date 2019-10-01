""" Common util file
"""


import numpy as np
import numpy.random as npr
import os
import skimage.io
import skimage.transform
import time
import pdb
import networkx as nx
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import random
import skimage.draw

import _init_paths
from config import cfg

STR_ELEM = np.array([[1,1,1],[1,1,0],[0,0,0]], dtype=np.bool) # eight-neighbors
#STR_ELEM = np.array([[0,1,0],[1,1,0],[0,0,0]], dtype=np.bool) # four-neighbors

# graph visualization
VIS_FIG_SIZE = (10,10)
VIS_NODE_SIZE = 50
VIS_ALPHA = 0.5 # (both for nodes and edges)
VIS_NODE_COLOR = ['b','r','y','g'] # tp/fp/fn(+tn)/tn
VIS_EDGE_COLOR = ['b','g','r'] # tp/fn/fp

DEBUG = False


class DataLayer(object):

    def __init__(self, db, is_training, use_padding=False):
        """Set the db to be used by this layer."""
        self._db = db
        self._is_training = is_training
        self._use_padding = use_padding
        if self._is_training:
            self._shuffle_db_inds()
        else:
            self._db_inds()

    def _shuffle_db_inds(self):
        """Randomly permute the db."""
        self._perm = np.random.permutation(np.arange(len(self._db)))
        self._cur = 0
        
    def _db_inds(self):
        """Permute the db."""
        self._perm = np.arange(len(self._db))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the db indices for the next minibatch."""
        cur_batch_size = cfg.TRAIN.BATCH_SIZE
        if self._is_training:
            if self._cur + cfg.TRAIN.BATCH_SIZE > len(self._db):
                self._shuffle_db_inds()    
        else:
            rem = len(self._db) - self._cur
            if rem >= cfg.TRAIN.BATCH_SIZE:
                cur_batch_size = cfg.TRAIN.BATCH_SIZE
            else:
                cur_batch_size = rem
        
        db_inds = self._perm[self._cur:self._cur + cur_batch_size]
        self._cur += cur_batch_size
        if (not self._is_training) and (self._cur>=len(self._db)):
            self._db_inds()
                
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in db_inds]
        return minibatch_db, get_minibatch(minibatch_db, self._is_training, \
                                           use_padding=self._use_padding)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        img_list, blobs = self._get_next_minibatch()
        return img_list, blobs

        
class GraphDataLayer(object):

    def __init__(self, db, is_training, \
                 edge_type='srns_geo_dist_binary', \
                 win_size=8, edge_geo_dist_thresh=20):
        """Set the db to be used by this layer."""
        self._db = db
        self._is_training = is_training
        self._edge_type = edge_type
        self._win_size = win_size
        self._edge_geo_dist_thresh = edge_geo_dist_thresh
        if self._is_training:
            self._shuffle_db_inds()
        else:
            self._db_inds()

    def _shuffle_db_inds(self):
        """Randomly permute the db."""
        self._perm = np.random.permutation(np.arange(len(self._db)))
        self._cur = 0
        
    def _db_inds(self):
        """Permute the db."""
        self._perm = np.arange(len(self._db))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the db indices for the next minibatch."""
        cur_batch_size = cfg.TRAIN.GRAPH_BATCH_SIZE
        if self._is_training:
            if self._cur + cfg.TRAIN.GRAPH_BATCH_SIZE > len(self._db):
                self._shuffle_db_inds()    
        else:
            rem = len(self._db) - self._cur
            if rem >= cfg.TRAIN.GRAPH_BATCH_SIZE:
                cur_batch_size = cfg.TRAIN.GRAPH_BATCH_SIZE
            else:
                cur_batch_size = rem
        
        db_inds = self._perm[self._cur:self._cur + cur_batch_size]
        self._cur += cur_batch_size
        if (not self._is_training) and (self._cur>=len(self._db)):
            self._db_inds()
                
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch."""
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in db_inds]
        return minibatch_db, get_minibatch(minibatch_db, self._is_training, \
                                           is_about_graph=True, \
                                           edge_type=self._edge_type, \
                                           win_size=self._win_size, \
                                           edge_geo_dist_thresh=self._edge_geo_dist_thresh)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        img_list, blobs = self._get_next_minibatch()
        return img_list, blobs
    
    def reinit(self, db, is_training, \
               edge_type='srns_geo_dist_binary', \
               win_size=8, edge_geo_dist_thresh=20):
        """Reinitialize with new arguments."""
        self._db = db
        self._is_training = is_training
        self._edge_type = edge_type
        self._win_size = win_size
        self._edge_geo_dist_thresh = edge_geo_dist_thresh
        if self._is_training:
            self._shuffle_db_inds()
        else:
            self._db_inds()


def get_minibatch(minibatch_db, is_training, \
                  is_about_graph=False, \
                  edge_type='srns_geo_dist_binary', \
                  win_size=8, edge_geo_dist_thresh=20, \
                  use_padding=False):
    """Given a minibatch_db, construct a blob."""

    if not is_about_graph:            
        im_blob, label_blob, fov_blob = _get_image_fov_blob(minibatch_db, is_training, use_padding=use_padding)
        blobs = {'img': im_blob, 'label': label_blob, 'fov': fov_blob}
 
    else:
        im_blob, label_blob, fov_blob, probmap_blob, \
        all_union_graph, \
        num_of_nodes_list, vec_aug_on, rot_angle = \
        _get_graph_fov_blob(minibatch_db, is_training, edge_type, win_size, edge_geo_dist_thresh)
        
        blobs = {'img': im_blob, 'label': label_blob, 'fov': fov_blob, 'probmap': probmap_blob,
                 'graph': all_union_graph,
                 'num_of_nodes_list': num_of_nodes_list,
                 'vec_aug_on': vec_aug_on,
                 'rot_angle': rot_angle}

    return blobs


def _get_image_fov_blob(minibatch_db, is_training, use_padding=False):
    """Builds an input blob from the images in the minibatch_db."""
    
    num_images = len(minibatch_db)
    processed_ims = []
    processed_labels = []
    processed_fovs = []
    if 'DRIVE' in minibatch_db[0]:
        im_ext = '_image.tif'
        label_ext = '_label.gif'
        fov_ext = '_mask.gif'
        pixel_mean = cfg.PIXEL_MEAN_DRIVE
        len_y = 592
        len_x = 592
    elif 'STARE' in minibatch_db[0]:
        im_ext = '.ppm'
        label_ext = '.ah.ppm'
        fov_ext = '_mask.png'
        pixel_mean = cfg.PIXEL_MEAN_STARE
        len_y = 704
        len_x = 704
    elif 'CHASE_DB1' in minibatch_db[0]:
        im_ext = '.jpg'
        label_ext = '_1stHO.png'
        fov_ext = '_mask.tif'
        pixel_mean = cfg.PIXEL_MEAN_CHASE_DB1
        len_y = 1024
        len_x = 1024
    elif 'HRF' in minibatch_db[0]:
        im_ext = '.bmp'
        label_ext = '.tif'
        fov_ext = '_mask.tif'
        pixel_mean = cfg.PIXEL_MEAN_HRF
        len_y = 768
        len_x = 768
        
    for i in xrange(num_images):
        im = skimage.io.imread(minibatch_db[i]+im_ext)
        label = skimage.io.imread(minibatch_db[i]+label_ext)
        label = label.reshape((label.shape[0],label.shape[1],1))
        fov = skimage.io.imread(minibatch_db[i]+fov_ext)
        if fov.ndim==2:
            fov = fov.reshape((fov.shape[0],fov.shape[1],1))
        else:
            fov = fov[:,:,[0]]
            
        if use_padding:
            temp = np.copy(im)
            im = np.zeros((len_y,len_x,3), dtype=temp.dtype)
            im[:temp.shape[0],:temp.shape[1],:] = temp
            temp = np.copy(label)
            label = np.zeros((len_y,len_x,1), dtype=temp.dtype)
            label[:temp.shape[0],:temp.shape[1],:] = temp
            temp = np.copy(fov)
            fov = np.zeros((len_y,len_x,1), dtype=temp.dtype)
            fov[:temp.shape[0],:temp.shape[1],:] = temp
        
        processed_im, processed_label, processed_fov, _ = \
        prep_im_fov_for_blob(im, label, fov, pixel_mean, is_training)
        processed_ims.append(processed_im)
        processed_labels.append(processed_label)
        processed_fovs.append(processed_fov)

    # Create a blob to hold the input images & labels & fovs
    im_blob = im_list_to_blob(processed_ims)
    label_blob = im_list_to_blob(processed_labels)
    fov_blob = im_list_to_blob(processed_fovs)

    return im_blob, label_blob, fov_blob


def _get_graph_fov_blob(minibatch_db, is_training, edge_type='srns_geo_dist_binary', \
                        win_size=8, edge_geo_dist_thresh=20):
    """Builds an input blob from the graphs in the minibatch_db."""
    
    num_graphs = len(minibatch_db)
    processed_ims = [] # image related
    processed_labels = [] # image related
    processed_fovs = [] # image related
    processed_probmaps = [] # image related
    all_graphs = [] # graph related
    num_of_nodes_list = [] # graph related
    
    # to apply the same aug in a mini-batch #
    if num_graphs > 1:
        given_aug_vec = np.zeros((7,), dtype=np.bool)
        if cfg.TRAIN.USE_LR_FLIPPED and npr.random_sample() >= 0.5:
            given_aug_vec[0] = True
        if cfg.TRAIN.USE_UD_FLIPPED and npr.random_sample() >= 0.5:
            given_aug_vec[1] = True
        if cfg.TRAIN.USE_ROTATION:
            given_aug_vec[2] = True
        if cfg.TRAIN.USE_SCALING:
            given_aug_vec[3] = True
        if cfg.TRAIN.USE_CROPPING:
            given_aug_vec[4] = True
        if cfg.TRAIN.USE_BRIGHTNESS_ADJUSTMENT:
            given_aug_vec[5] = True
        if cfg.TRAIN.USE_CONTRAST_ADJUSTMENT:
            given_aug_vec[6] = True
    # to apply the same aug in a mini-batch #
    
    if 'DRIVE' in minibatch_db[0]:
        im_root_path = '../DRIVE/all'
        im_ext = '_image.tif'
        label_ext = '_label.gif'
        fov_ext = '_mask.gif'
        pixel_mean = cfg.PIXEL_MEAN_DRIVE
        len_y = 592
        len_x = 592
    elif 'STARE' in minibatch_db[0]:
        im_root_path = '../STARE/all'
        im_ext = '.ppm'
        label_ext = '.ah.ppm'
        fov_ext = '_mask.png'
        pixel_mean = cfg.PIXEL_MEAN_STARE
        len_y = 704
        len_x = 704
    elif 'CHASE' in minibatch_db[0]:
        im_root_path = '../CHASE_DB1/all'
        im_ext = '.jpg'
        label_ext = '_1stHO.png'
        fov_ext = '_mask.tif'
        pixel_mean = cfg.PIXEL_MEAN_CHASE_DB1
        len_y = 1024
        len_x = 1024
    elif 'HRF' in minibatch_db[0]:
        im_root_path = '../HRF/all_768'
        im_ext = '.bmp'
        label_ext = '.tif'
        fov_ext = '_mask.tif'
        pixel_mean = cfg.PIXEL_MEAN_HRF
        len_y = 768
        len_x = 768
    for i in xrange(num_graphs):
        
        # load images
        cur_path = minibatch_db[i]
        cur_name = cur_path[find(cur_path,'/')[-1]+1:]
        
        im = skimage.io.imread(os.path.join(im_root_path, cur_name+im_ext))
        label = skimage.io.imread(os.path.join(im_root_path, cur_name+label_ext))      
        label = label.reshape((label.shape[0],label.shape[1],1))       
        fov = skimage.io.imread(os.path.join(im_root_path, cur_name+fov_ext))
        if fov.ndim==2:
            fov = fov.reshape((fov.shape[0],fov.shape[1],1))
        else:
            fov = fov[:,:,[0]]
        probmap = skimage.io.imread(cur_path+'_prob.png') # cnn results will be used for loss masking
        probmap = probmap.reshape((probmap.shape[0],probmap.shape[1],1))
        
        temp = np.copy(im)
        im = np.zeros((len_y,len_x,3), dtype=temp.dtype)
        im[:temp.shape[0],:temp.shape[1],:] = temp
        temp = np.copy(label)
        label = np.zeros((len_y,len_x,1), dtype=temp.dtype)
        label[:temp.shape[0],:temp.shape[1],:] = temp
        temp = np.copy(fov)
        fov = np.zeros((len_y,len_x,1), dtype=temp.dtype)
        fov[:temp.shape[0],:temp.shape[1],:] = temp
        temp = np.copy(probmap)
        probmap = np.zeros((len_y,len_x,1), dtype=temp.dtype)
        probmap[:temp.shape[0],:temp.shape[1],:] = temp
        
        # load graphs
        if 'srns' not in edge_type:
            raise NotImplementedError
        else:
            win_size_str = '_%.2d_%.2d'%(win_size,edge_geo_dist_thresh)       
            graph = nx.read_gpickle(cur_path+win_size_str+'.graph_res')
            
        union_graph = nx.convert_node_labels_to_integers(graph)
        n_nodes_in_graph = union_graph.number_of_nodes()
        node_idx_map = np.zeros(im.shape[:2])
        for j in xrange(n_nodes_in_graph):
            node_idx_map[union_graph.nodes[j]['y'],union_graph.nodes[j]['x']] = j+1
        
        if num_graphs > 1: # not used
            raise NotImplementedError
        else:
            processed_im, processed_label, processed_fov, processed_probmap, processed_node_idx_map, \
            vec_aug_on, (crop_y1,crop_y2,crop_x1,crop_x2), rot_angle = \
            prep_im_label_fov_probmap_for_blob(im, label, fov, probmap, node_idx_map, pixel_mean, is_training, win_size)
        
        processed_ims.append(processed_im)
        processed_labels.append(processed_label)
        processed_fovs.append(processed_fov)
        processed_probmaps.append(processed_probmap)
        
        node_ys, node_xs = np.where(processed_node_idx_map)
        for j in xrange(len(node_ys)):
            cur_node_idx = processed_node_idx_map[node_ys[j],node_xs[j]] 
            union_graph.nodes[cur_node_idx-1]['y'] = node_ys[j]
            union_graph.nodes[cur_node_idx-1]['x'] = node_xs[j]
        union_graph = nx.convert_node_labels_to_integers(union_graph) 
        n_nodes_in_graph = union_graph.number_of_nodes()
        
        """if vec_aug_on[0]:
            for j in xrange(n_nodes_in_graph):
                union_graph.nodes[j]['x'] = label.shape[1]-union_graph.nodes[j]['x']-1
                
        if vec_aug_on[1]:
            for j in xrange(n_nodes_in_graph):
                union_graph.nodes[j]['y'] = label.shape[0]-union_graph.nodes[j]['y']-1"""            
        
        if vec_aug_on[4]:
            del_node_list = []
            for j in xrange(n_nodes_in_graph):
                if (union_graph.nodes[j]['y']>=crop_y1 and \
                    union_graph.nodes[j]['y']<crop_y2 and \
                    union_graph.nodes[j]['x']>=crop_x1 and \
                    union_graph.nodes[j]['x']<crop_x2):
                    union_graph.nodes[j]['y'] = union_graph.nodes[j]['y']-crop_y1
                    union_graph.nodes[j]['x'] = union_graph.nodes[j]['x']-crop_x1
                else:
                    del_node_list.append(j)       
            union_graph.remove_nodes_from(del_node_list)
            union_graph = nx.convert_node_labels_to_integers(union_graph) 
            n_nodes_in_graph = union_graph.number_of_nodes()
            
        n_nodes_in_graph_fp = 0
            
        if DEBUG:
            # visualize the constructed graph
            visualize_graph(processed_im, union_graph, show_graph=False, \
                            save_graph=True, num_nodes_each_type=[n_nodes_in_graph,n_nodes_in_graph_fp], 
                            save_path='graph_union.png')
        
        all_graphs.append(union_graph)
        num_of_nodes_list.append(n_nodes_in_graph+n_nodes_in_graph_fp)

    # Create a blob to hold the input images & labels & graphs
    im_blob = im_list_to_blob(processed_ims)
    label_blob = im_list_to_blob(processed_labels)
    fov_blob = im_list_to_blob(processed_fovs)
    probmap_blob = im_list_to_blob(processed_probmaps)
    all_union_graph = nx.algorithms.operators.all.disjoint_union_all(all_graphs)    
    if DEBUG:
        num_nodes_list = map(lambda x: x.number_of_nodes(), all_graphs)
        assert all_union_graph.number_of_nodes()==np.sum(np.array(num_nodes_list))
        num_edges_list = map(lambda x: x.number_of_edges(), all_graphs)
        assert all_union_graph.number_of_edges()==np.sum(np.array(num_edges_list))

    return im_blob, label_blob, fov_blob, probmap_blob, all_union_graph, num_of_nodes_list, vec_aug_on, rot_angle


def prep_im_fov_for_blob(im, label, fov, pixel_mean, is_training):
    """Preprocess images for use in a blob."""

    im = im.astype(np.float32, copy=False)/255.
    label = label.astype(np.float32, copy=False)/255.        
    fov = fov.astype(np.float32, copy=False)/255.
    
    #skimage.io.imsave('img.bmp', (im*255).astype(int))
    #skimage.io.imsave('label.bmp', (label.reshape(label.shape[0],label.shape[1])*255).astype(int))
    
    vec_aug_on = np.zeros((7,), dtype=np.bool)

    if is_training:
        if cfg.TRAIN.USE_LR_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[0] = True
            im = im[:, ::-1, :]
            label = label[:, ::-1, :]
            fov = fov[:, ::-1, :]
        
        if cfg.TRAIN.USE_UD_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[1] = True
            im = im[::-1, :, :]
            label = label[::-1, :, :]
            fov = fov[::-1, :, :]
            
        if cfg.TRAIN.USE_ROTATION:
            vec_aug_on[2] = True
            rot_angle = np.random.uniform(-cfg.TRAIN.ROTATION_MAX_ANGLE,cfg.TRAIN.ROTATION_MAX_ANGLE)
            """im_r = skimage.transform.rotate(im[:,:,0], rot_angle, cval=0.)
            im_g = skimage.transform.rotate(im[:,:,1], rot_angle, cval=0.)
            im_b = skimage.transform.rotate(im[:,:,2], rot_angle, cval=0.)"""
            im_r = skimage.transform.rotate(im[:,:,0], rot_angle, cval=pixel_mean[0]/255.)
            im_g = skimage.transform.rotate(im[:,:,1], rot_angle, cval=pixel_mean[1]/255.)
            im_b = skimage.transform.rotate(im[:,:,2], rot_angle, cval=pixel_mean[2]/255.)
            im = np.dstack((im_r,im_g,im_b))
            label = skimage.transform.rotate(label, rot_angle, cval=0., order=0)
            fov = skimage.transform.rotate(fov, rot_angle, cval=0., order=0)
        
        if cfg.TRAIN.USE_SCALING:
            vec_aug_on[3] = True
            scale = np.random.uniform(cfg.TRAIN.SCALING_RANGE[0],cfg.TRAIN.SCALING_RANGE[1])
            im = skimage.transform.rescale(im, scale)
            label = skimage.transform.rescale(label, scale, order=0)
            fov = skimage.transform.rescale(fov, scale, order=0)
                        
        if cfg.TRAIN.USE_CROPPING:
            vec_aug_on[4] = True
            cur_h = np.random.random_integers(im.shape[0]*0.5,im.shape[0]*0.8)
            cur_w = np.random.random_integers(im.shape[1]*0.5,im.shape[1]*0.8)
            cur_y1 = np.random.random_integers(0,im.shape[0]-cur_h)
            cur_x1 = np.random.random_integers(0,im.shape[1]-cur_w)
            cur_y2 = cur_y1 + cur_h
            cur_x2 = cur_x1 + cur_w
            im = im[cur_y1:cur_y2,cur_x1:cur_x2,:]
            label = label[cur_y1:cur_y2,cur_x1:cur_x2,:]
            fov = fov[cur_y1:cur_y2,cur_x1:cur_x2,:]
        
        if cfg.TRAIN.USE_BRIGHTNESS_ADJUSTMENT:
            vec_aug_on[5] = True
            im += np.random.uniform(-cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA,cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA)
            im = np.clip(im, 0, 1)  
    
        if cfg.TRAIN.USE_CONTRAST_ADJUSTMENT:
            vec_aug_on[6] = True
            mm = np.mean(im)
            im = (im-mm)*np.random.uniform(cfg.TRAIN.CONTRAST_ADJUSTMENT_LOWER_FACTOR,cfg.TRAIN.CONTRAST_ADJUSTMENT_UPPER_FACTOR) + mm                         
            im = np.clip(im, 0, 1)    

    #skimage.io.imsave('img_1.bmp', (im*255).astype(int))

    # original
    im -= np.array(pixel_mean)/255.
    im = im*255.
    
    """# contrast enhancement
    im_f = skimage.filters.gaussian(im, sigma=10, multichannel=True)
    im = im-im_f
    im = im*255."""
    
    label = label>=0.5
    fov = fov>=0.5
    
    #skimage.io.imsave('label_1.bmp', (label.reshape(label.shape[0],label.shape[1])*255).astype(int))
    
    return im, label, fov, vec_aug_on


def prep_im_label_fov_probmap_for_blob(im, label, fov, probmap, node_idx_map, pixel_mean, is_training, win_size):
    """Preprocess images for use in a blob."""

    im = im.astype(np.float32, copy=False)/255.
    label = label.astype(np.float32, copy=False)/255.        
    fov = fov.astype(np.float32, copy=False)/255.  
    probmap = probmap.astype(np.float32, copy=False)/255.

    vec_aug_on = np.zeros((7,), dtype=np.bool)

    cur_y1 = 0  
    cur_y2 = 0                      
    cur_x1 = 0
    cur_x2 = 0
    rot_angle = 0
    if is_training:
        if cfg.TRAIN.USE_LR_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[0] = True
            im = im[:, ::-1, :]
            label = label[:, ::-1, :]
            fov = fov[:, ::-1, :]
            probmap = probmap[:, ::-1, :]
            node_idx_map = node_idx_map[:, ::-1]
        
        if cfg.TRAIN.USE_UD_FLIPPED and npr.random_sample() >= 0.5:
            vec_aug_on[1] = True
            im = im[::-1, :, :]
            label = label[::-1, :, :]
            fov = fov[::-1, :, :]
            probmap = probmap[::-1, :, :]
            node_idx_map = node_idx_map[::-1, :]
            
        if cfg.TRAIN.USE_ROTATION:
            vec_aug_on[2] = True
            
            len_ori_y,len_ori_x = im.shape[:2]
            
            rot_angle = np.random.choice([0,90,180,270])
            im_r = skimage.transform.rotate(im[:,:,0], rot_angle, cval=pixel_mean[0]/255., resize=True)
            im_g = skimage.transform.rotate(im[:,:,1], rot_angle, cval=pixel_mean[1]/255., resize=True)
            im_b = skimage.transform.rotate(im[:,:,2], rot_angle, cval=pixel_mean[2]/255., resize=True)
            im = np.dstack((im_r,im_g,im_b))
            label = skimage.transform.rotate(label, rot_angle, cval=0., order=0, resize=True)
            fov = skimage.transform.rotate(fov, rot_angle, cval=0., order=0, resize=True)
            probmap = skimage.transform.rotate(probmap, rot_angle, cval=0., resize=True)
            node_idx_map = skimage.transform.rotate(node_idx_map, rot_angle, cval=0., order=0, resize=True)
            
            im = im[:len_ori_y,:len_ori_x,:]
            label = label[:len_ori_y,:len_ori_x,:]
            fov = fov[:len_ori_y,:len_ori_x,:]
            probmap = probmap[:len_ori_y,:len_ori_x,:]
            node_idx_map = node_idx_map[:len_ori_y,:len_ori_x]
            
        if cfg.TRAIN.USE_SCALING:
            vec_aug_on[3] = True
            scale = np.random.uniform(cfg.TRAIN.SCALING_RANGE[0],cfg.TRAIN.SCALING_RANGE[1])
            im = skimage.transform.rescale(im, scale)
            label = skimage.transform.rescale(label, scale, order=0)
            fov = skimage.transform.rescale(fov, scale, order=0)
            probmap = skimage.transform.rescale(probmap, scale)
            node_idx_map = skimage.transform.rescale(node_idx_map, scale, order=0)
        
        if cfg.TRAIN.USE_CROPPING:
            vec_aug_on[4] = True
            
            # cropping dependent on 'win_size'
            cur_h = (np.random.random_integers(im.shape[0]*0.5,im.shape[0]*0.8)//win_size)*win_size
            cur_w = (np.random.random_integers(im.shape[1]*0.5,im.shape[1]*0.8)//win_size)*win_size
            if vec_aug_on[0]:
                cur_y1 = np.random.choice(range(im.shape[0]%win_size,im.shape[0]-cur_h,win_size))
                cur_x1 = np.random.choice(range(im.shape[1]%win_size,im.shape[1]-cur_w,win_size))
            else:
                cur_y1 = np.random.choice(range(0,im.shape[0]-cur_h,win_size))
                cur_x1 = np.random.choice(range(0,im.shape[1]-cur_w,win_size))
            cur_y2 = cur_y1 + cur_h
            cur_x2 = cur_x1 + cur_w
            
            im = im[cur_y1:cur_y2,cur_x1:cur_x2,:]
            label = label[cur_y1:cur_y2,cur_x1:cur_x2,:]
            fov = fov[cur_y1:cur_y2,cur_x1:cur_x2,:]
            probmap = probmap[cur_y1:cur_y2,cur_x1:cur_x2,:]
            node_idx_map = node_idx_map[cur_y1:cur_y2,cur_x1:cur_x2]
        
        if cfg.TRAIN.USE_BRIGHTNESS_ADJUSTMENT:
            vec_aug_on[5] = True
            im += np.random.uniform(-cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA,cfg.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA)
            im = np.clip(im, 0, 1)  
    
        if cfg.TRAIN.USE_CONTRAST_ADJUSTMENT:
            vec_aug_on[6] = True
            mm = np.mean(im)
            im = (im-mm)*np.random.uniform(cfg.TRAIN.CONTRAST_ADJUSTMENT_LOWER_FACTOR,cfg.TRAIN.CONTRAST_ADJUSTMENT_UPPER_FACTOR) + mm                         
            im = np.clip(im, 0, 1)    

    # original
    im -= np.array(pixel_mean)/255.
    im = im*255.
    
    """# contrast enhancement
    im_f = skimage.filters.gaussian(im, sigma=10, multichannel=True)
    im = im-im_f
    im = im*255."""
    
    label = label>=0.5
    fov = fov>=0.5
    
    return im, label, fov, probmap, node_idx_map, vec_aug_on, (cur_y1,cur_y2,cur_x1,cur_x2), rot_angle
    
    
def im_list_to_blob(ims):
    """Convert a list of images into a network input."""
    
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], max_shape[2]),
                    dtype=ims[0].dtype)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]            
    

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


# append self connections (diagonal terms in adjacency matrix) and binarize
def preprocess_graph_gat(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    return indices, adj.data, adj.shape


def visualize_graph(im, graph, show_graph=False, save_graph=True, \
                    num_nodes_each_type=None, custom_node_color=None, \
                    tp_edges=None, fn_edges=None, fp_edges=None, \
                    save_path='graph.png'):
       
    plt.figure(figsize=VIS_FIG_SIZE)
    if im.dtype==np.bool:
        bg = im.astype(int)*255
    else:
        bg = im
    
    if len(bg.shape)==2:
        plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
    elif len(bg.shape)==3:
        plt.imshow(bg)
    #plt.imshow(bg, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    pos = {}
    node_list = list(graph.nodes)
    for i in node_list:
        pos[i] = [graph.nodes[i]['x'],graph.nodes[i]['y']]
    
    if custom_node_color is not None:
        node_color = custom_node_color
    else:
        if num_nodes_each_type is None:
            node_color = 'b'
        else:
            if not (graph.number_of_nodes()==np.sum(num_nodes_each_type)):
                raise ValueError('Wrong number of nodes')
            node_color = [VIS_NODE_COLOR[0]]*num_nodes_each_type[0] + [VIS_NODE_COLOR[1]]*num_nodes_each_type[1] 

    nx.draw(graph, pos, node_color='green', edge_color='blue', width=1, node_size=10, alpha=VIS_ALPHA)
    #nx.draw(graph, pos, node_color='darkgreen', edge_color='black', width=3, node_size=30, alpha=VIS_ALPHA)
    #nx.draw(graph, pos, node_color=node_color, node_size=VIS_NODE_SIZE, alpha=VIS_ALPHA)
    
    if tp_edges is not None:
        nx.draw_networkx_edges(graph, pos,
                               edgelist=tp_edges,
                               width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[0])
    if fn_edges is not None:
        nx.draw_networkx_edges(graph, pos,
                               edgelist=fn_edges,
                               width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[1])
    if fp_edges is not None:
        nx.draw_networkx_edges(graph, pos,
                               edgelist=fp_edges,
                               width=3, alpha=VIS_ALPHA, edge_color=VIS_EDGE_COLOR[2])

    if save_graph:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show_graph:
        plt.show()
    
    plt.cla()
    plt.clf()
    plt.close()
    

def get_auc_ap_score(labels, preds):

    auc_score = roc_auc_score(labels, preds)
    ap_score = average_precision_score(labels, preds)

    return auc_score, ap_score


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def get_node_byx_from_graph(graph, num_of_nodes_list):
    node_byxs = np.zeros((graph.number_of_nodes(),3), dtype=np.int32)
    node_idx = 0
    for sub_graph_idx, cur_num_nodes in enumerate(num_of_nodes_list):
        for i in range(node_idx,node_idx+cur_num_nodes):
            node_byxs[i,:] = [sub_graph_idx,graph.nodes[i]['y'],graph.nodes[i]['x']]
        node_idx = node_idx+cur_num_nodes
    
    return node_byxs