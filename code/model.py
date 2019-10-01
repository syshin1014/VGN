""" Common model file
"""


import numpy as np
import cPickle
import pdb
import tensorflow as tf

from config import cfg


DEBUG = False


def _activation_summary(x_name, x):
    """Helper to create summaries for activations.

    #Creates a summary that shows activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
        x_name: Tensor name
        x: Tensor
    Returns:
        nothing
    """

    tf.summary.histogram(x_name + '/activations', x)
    tf.summary.scalar(x_name + '/sparsity', tf.nn.zero_fraction(x))


# https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py
# https://gist.github.com/akiross/754c7b87a2af8603da78b46cdaaa5598
def get_deconv_filter(f_shape):
    # f_shape = [ksize, ksize, out_features, in_features]
    width = f_shape[0]
    height = f_shape[0]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
        
    return weights


def add_tensors_wo_none(tensor_list):
    # Adds all input tensors element-wise while filtering out none tensors.
    # print tensor_list
    temp_list = filter(lambda x: (x is not None), tensor_list)
    if len(temp_list):
        return tf.add_n(temp_list)
    else:
        return None
     

class base_model():
    def __init__(self, weight_file_path):

        if weight_file_path is not None:
            with open(weight_file_path) as f:
                self.pretrained_weights = cPickle.load(f)       
    
    def new_conv_layer(self, bottom, filter_shape, stride=[1,1,1,1], init=tf.truncated_normal_initializer(0., 0.01), \
                       norm_type=None, use_relu=False, is_training=True, name=None):
        
        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=init)
            out = tf.nn.conv2d(bottom, w, stride, padding='SAME')
            
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.USE_BRN)
            elif norm_type=='GN':
                out = self.group_norm(out, num_group=min(cfg.GN_MIN_NUM_G, filter_shape[-1]/cfg.GN_MIN_CHS_PER_G))
            else:
                b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))
                out = tf.nn.bias_add(out, b)

            if use_relu:
                out = tf.nn.relu(out)

        return out
    
    def new_fc_layer(self, bottom, input_size, output_size, init=tf.truncated_normal_initializer(0., 0.01), \
                     norm_type=None, use_relu=False, is_training=True, name=None):
        shape = bottom.get_shape().as_list()
        dim = np.prod(shape[1:])
        x = tf.reshape(bottom, [-1,dim])

        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=init)
            out = tf.matmul(x, w)
            
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.USE_BRN)
            elif norm_type=='GN':
                out = self.group_norm(out, num_group=min(cfg.GN_MIN_NUM_G, output_size/cfg.GN_MIN_CHS_PER_G))
            else:
                b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
                out = out+b
                
            if use_relu:
                out = tf.nn.relu(out)

        return out
    
    def new_deconv_layer(self, bottom, filter_shape, output_shape, strides, norm_type=None, use_relu=False, is_training=True, name=None):
        weights = get_deconv_filter(filter_shape)       
        
        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                    "W",
                    shape=weights.shape,
                    initializer=tf.constant_initializer(value=weights,dtype=tf.float32))

            out = tf.nn.conv2d_transpose(bottom, w, output_shape, strides, padding='SAME')
            
            if DEBUG:
                out = tf.Print(out, [tf.shape(out)],
                               message='Shape of %s' % name,
                               summarize=4, first_n=1)
                
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.USE_BRN)
            elif norm_type=='GN':
                out = self.group_norm(out, num_group=min(cfg.GN_MIN_NUM_G, filter_shape[-2]/cfg.GN_MIN_CHS_PER_G))
            else:
                b = tf.get_variable(
                    "b",
                    shape=weights.shape[-2],
                    initializer=tf.constant_initializer(0.))
                out = tf.nn.bias_add(out, b)

            if use_relu:
                out = tf.nn.relu(out)    

        return out

    # https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/pool.py
    # https://github.com/tensorflow/tensorflow/issues/2169        
    def unpool(self, pool, ind, ksize, name):
        """
        Unpooling layer after max_pool_with_argmax.
        Args :
            pool : max pooled output tensor
            ind : argmax indices
            ksize : ksize is the same as for the pool
        Return :
            ret : unpooled tensor
        """
        with tf.variable_scope(name) as scope:
            input_shape =  tf.shape(pool)
            output_shape = [input_shape[0], input_shape[1]*ksize[1], input_shape[2]*ksize[2], input_shape[3]]
    
            flat_input_size = tf.cumprod(input_shape)[-1]
            flat_output_shape = tf.stack([output_shape[0], output_shape[1]*output_shape[2]*output_shape[3]])
    
            pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
            batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                              shape=tf.stack([input_shape[0], 1, 1, 1]))
            b = tf.ones_like(ind)*batch_range
            b = tf.reshape(b, tf.stack([flat_input_size, 1]))
            ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
            ind_ = tf.concat([b, ind_], 1)
    
            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, tf.stack(output_shape))
        
        return ret
    
    def group_norm(self, input, num_group=32, epsilon=1e-05):
        # We here assume the channel-last ordering (NHWC)
        
        num_ch = input.get_shape().as_list()[-1]
        num_group = min(num_group, num_ch)
        
        NHWCG = tf.concat([tf.slice(tf.shape(input),[0],[3]), tf.constant([num_ch//num_group, num_group])], axis=0)
        output = tf.reshape(input, NHWCG)
        
        mean, var = tf.nn.moments(output, [1, 2, 3], keep_dims=True)
        output = (output - mean) / tf.sqrt(var + epsilon)
        
        # gamma and beta
        gamma = tf.get_variable('gamma', [1, 1, 1, num_ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, num_ch], initializer=tf.constant_initializer(0.0))
    
        output = tf.reshape(output, tf.shape(input)) * gamma + beta
        
        return output
    
    def group_norm_fc(self, input, num_group=32, epsilon=1e-05):
        # We here assume the channel-last ordering (NHWC)
        
        num_ch = input.get_shape().as_list()[-1]
        num_group = min(num_group, num_ch)
        
        NCG = tf.concat([tf.slice(tf.shape(input),[0],[1]), tf.constant([num_ch//num_group, num_group])], axis=0)
        output = tf.reshape(input, NCG)
        
        mean, var = tf.nn.moments(output, [1], keep_dims=True)
        output = (output - mean) / tf.sqrt(var + epsilon)
        
        # gamma and beta
        gamma = tf.get_variable('gamma', [1, num_ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, num_ch], initializer=tf.constant_initializer(0.0))
    
        output = tf.reshape(output, tf.shape(input)) * gamma + beta
        
        return output
    
    def group_norm_layer(self, input, num_group=32, epsilon=1e-05, name=None):
        # We here assume the channel-last ordering (NHWC)
        
        with tf.variable_scope(name) as scope:
        
            num_ch = input.get_shape().as_list()[-1]
            num_group = min(num_group, num_ch)
            
            NHWCG = tf.concat([tf.slice(tf.shape(input),[0],[3]), tf.constant([num_ch//num_group, num_group])], axis=0)
            output = tf.reshape(input, NHWCG)
            
            mean, var = tf.nn.moments(output, [1, 2, 3], keep_dims=True)
            output = (output - mean) / tf.sqrt(var + epsilon)
            
            # gamma and beta
            gamma = tf.get_variable('gamma', [1, 1, 1, num_ch], initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [1, 1, 1, num_ch], initializer=tf.constant_initializer(0.0))
        
            output = tf.reshape(output, tf.shape(input)) * gamma + beta
        
        return output


class vessel_segm_cnn(base_model):
    def __init__(self, params, weight_file_path):
        base_model.__init__(self, weight_file_path)
        self.params = params
        self.cnn_model = params.cnn_model
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.build_model()
        
    def build_model(self):
        """Build the model."""
        print("Building the model...")
        if self.cnn_model=='driu':
            self.build_driu()
        elif self.cnn_model=='driu_large':
            self.build_driu_large()
        else:
            pass
        print("Model built.")
        
    def build_driu(self):

        imgs = tf.placeholder(tf.float32, [None, None, None, 3], name='imgs')
        labels = tf.placeholder(tf.int64, [None, None, None, 1], name='labels')
        fov_masks = tf.placeholder(tf.int64, [None, None, None, 1], name='fov_masks')

        is_training = tf.placeholder(tf.bool, [])

        conv1_1 = self.new_conv_layer(imgs, [3,3,3,64], use_relu=True, name='conv1_1')
        _activation_summary('conv1_1', conv1_1)
        conv1_2 = self.new_conv_layer(conv1_1, [3,3,64,64], use_relu=True, name='conv1_2')
        _activation_summary('conv1_2', conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
                               
        conv2_1 = self.new_conv_layer(pool1, [3,3,64,128], use_relu=True, name='conv2_1')
        _activation_summary('conv2_1', conv2_1)
        conv2_2 = self.new_conv_layer(conv2_1, [3,3,128,128], use_relu=True, name='conv2_2')
        _activation_summary('conv2_2', conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')                               
                               
        conv3_1 = self.new_conv_layer(pool2, [3,3,128,256], use_relu=True, name='conv3_1')
        _activation_summary('conv3_1', conv3_1)
        conv3_2 = self.new_conv_layer(conv3_1, [3,3,256,256], use_relu=True, name='conv3_2')
        _activation_summary('conv3_2', conv3_2)
        conv3_3 = self.new_conv_layer(conv3_2, [3,3,256,256], use_relu=True, name='conv3_3')
        _activation_summary('conv3_3', conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
                               
        conv4_1 = self.new_conv_layer(pool3, [3,3,256,512], use_relu=True, name='conv4_1')
        _activation_summary('conv4_1', conv4_1)
        conv4_2 = self.new_conv_layer(conv4_1, [3,3,512,512], use_relu=True, name='conv4_2')
        _activation_summary('conv4_2', conv4_2)
        conv4_3 = self.new_conv_layer(conv4_2, [3,3,512,512], use_relu=True, name='conv4_3')
        _activation_summary('conv4_3', conv4_3)

        # specialized layers        
        num_ch = 16 # fixed
        
        target_shape = tf.concat(values=[tf.slice(tf.shape(imgs), [0], [3]),tf.constant(num_ch,shape=[1,])], axis=0)  
        spe_1 = self.new_conv_layer(conv1_2, [3,3,64,num_ch], use_relu=True, name='spe_1')
        _activation_summary('spe_1', spe_1)
        spe_2 = self.new_conv_layer(conv2_2, [3,3,128,num_ch], use_relu=True, name='spe_2')
        _activation_summary('spe_2', spe_2)
        resized_spe_2 = self.new_deconv_layer(spe_2, [4,4,num_ch,num_ch], target_shape, [1,2,2,1], use_relu=True, name='resized_spe_2') 
        spe_3 = self.new_conv_layer(conv3_3, [3,3,256,num_ch], use_relu=True, name='spe_3')
        _activation_summary('spe_3', spe_3)
        resized_spe_3 = self.new_deconv_layer(spe_3, [8,8,num_ch,num_ch], target_shape, [1,4,4,1], use_relu=True, name='resized_spe_3')
        spe_4 = self.new_conv_layer(conv4_3, [3,3,512,num_ch], use_relu=True, name='spe_4')
        _activation_summary('spe_4', spe_4)
        resized_spe_4 = self.new_deconv_layer(spe_4, [16,16,num_ch,num_ch], target_shape, [1,8,8,1], use_relu=True, name='resized_spe_4')  
        spe_concat = tf.concat(values=[spe_1,resized_spe_2,resized_spe_3,resized_spe_4], axis=3)

        output = self.new_conv_layer(spe_concat, [1,1,num_ch*4,1], name='output')
        _activation_summary('output', output)
        
        fg_prob = tf.sigmoid(output)

        ### Compute the loss ###
        binary_mask_fg = tf.to_float(tf.equal(labels, 1))
        binary_mask_bg = tf.to_float(tf.not_equal(labels, 1))
        combined_mask = tf.concat(values=[binary_mask_bg,binary_mask_fg], axis=3)
        flat_one_hot_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))
        flat_labels = tf.reshape(tensor=labels, shape=(-1,))
        flat_logits = tf.reshape(tensor=output, shape=(-1,))        
        cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=tf.to_float(flat_labels))
        
        """# weighted cross entropy loss (in/out fov)
        num_pixel = tf.size(labels)
        num_pixel_fg = tf.count_nonzero(binary_mask_fg, dtype=tf.int32)
        num_pixel_bg = num_pixel - num_pixel_fg
        class_weight = tf.cast(tf.concat(values=[tf.reshape(tf.divide(num_pixel_fg,num_pixel),(1,1)), \
                                                tf.reshape(tf.divide(num_pixel_bg,num_pixel),(1,1))], axis=1), dtype=tf.float32)
        weight_per_label = tf.transpose(tf.matmul(flat_one_hot_labels,tf.transpose(class_weight))) #shape [1, TRAIN.BATCH_SIZE]
        # this is the weight for each datapoint, depending on its label
        loss = tf.reduce_mean(tf.multiply(weight_per_label,cross_entropies))"""

        # weighted cross entropy loss (in fov)
        num_pixel = tf.reduce_sum(fov_masks)
        num_pixel_fg = tf.count_nonzero(binary_mask_fg, dtype=tf.int64)
        num_pixel_bg = num_pixel - num_pixel_fg
        class_weight = tf.cast(tf.concat(values=[tf.reshape(tf.divide(num_pixel_fg,num_pixel),(1,1)), \
                                                tf.reshape(tf.divide(num_pixel_bg,num_pixel),(1,1))], axis=1), dtype=tf.float32)
        weight_per_label = tf.transpose(tf.matmul(flat_one_hot_labels,tf.transpose(class_weight))) #shape [1, TRAIN.BATCH_SIZE]
        # this is the weight for each datapoint, depending on its label
        reshaped_fov_masks = tf.reshape(tensor=tf.cast(fov_masks, tf.float32), shape=(-1,))
        reshaped_fov_masks /= tf.reduce_mean(reshaped_fov_masks)
        loss = tf.reduce_mean(tf.multiply(tf.multiply(reshaped_fov_masks, weight_per_label), cross_entropies))
         
        weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * cfg.TRAIN.WEIGHT_DECAY_RATE
        loss += weight_decay
        
        ### Compute the accuracy ###
        flat_bin_output = tf.greater_equal(tf.reshape(tensor=fg_prob, shape=(-1,)), 0.5)
        # accuracy
        correct = tf.to_float(tf.equal(flat_bin_output,tf.cast(flat_labels, tf.bool)))
        accuracy = tf.reduce_mean(correct)
        # precision, recall
        num_fg_output = tf.reduce_sum(tf.to_float(flat_bin_output))
        tp = tf.reduce_sum(tf.to_float(tf.logical_and(tf.cast(flat_labels, dtype=tf.bool), flat_bin_output)))
        pre = tf.divide(tp,tf.add(num_fg_output,cfg.EPSILON))
        rec = tf.divide(tp,tf.to_float(num_pixel_fg))
        
        ### Build the solver ###
        if self.params.opt=='adam':
            train_op = tf.train.AdamOptimizer(self.params.lr, epsilon=0.1).minimize(loss, global_step=self.global_step)
        elif self.params.opt=='sgd':
            if self.params.lr_decay=='const':
                # constant
                optimizer = tf.train.MomentumOptimizer(self.params.lr, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                grads_and_vars = optimizer.compute_gradients(loss)
                grads_and_vars = map(lambda gv: (0.01*gv[0],gv[1]) if 'output' in gv[1].name else (gv[0],gv[1]), grads_and_vars)
                grads_and_vars = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), grads_and_vars)
                #grads_and_vars = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), grads_and_vars)
                #grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
            elif self.params.lr_decay=='pc':
                # piecewise_constant
                boundaries = [int(self.params.max_iters*0.5),int(self.params.max_iters*0.75)]
                values = [self.params.lr,self.params.lr*0.5,self.params.lr*0.25]
                learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
                #train_op = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True).minimize(loss, global_step=self.global_step)
                optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                grads_and_vars = optimizer.compute_gradients(loss)
                grads_and_vars = map(lambda gv: (0.01*gv[0],gv[1]) if 'output' in gv[1].name else (gv[0],gv[1]), grads_and_vars)
                grads_and_vars = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), grads_and_vars)
                #grads_and_vars = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), grads_and_vars)
                #grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
            elif self.params.lr_decay=='exp':
                # exponential_decay
                learning_rate = tf.train.exponential_decay(self.params.lr, self.global_step, self.params.max_iters/20, 0.9, staircase=False)
                #train_op = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True).minimize(loss, global_step=self.global_step)   
                optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                grads_and_vars = optimizer.compute_gradients(loss)
                grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        ### Hang up the results ###
        self.imgs = imgs
        self.labels = labels
        self.fov_masks = fov_masks
        self.is_training = is_training
        
        self.conv_feats = spe_concat
        self.output = output
        self.fg_prob = fg_prob
        
        self.flat_bin_output = flat_bin_output
        self.tp = tp
        self.num_fg_output = num_fg_output
        self.num_pixel_fg = num_pixel_fg        
        
        self.accuracy = accuracy
        self.precision = pre        
        self.recall = rec
        
        self.loss = loss
        self.train_op = train_op

        
    def build_driu_large(self):

        imgs = tf.placeholder(tf.float32, [None, None, None, 3], name='imgs')
        labels = tf.placeholder(tf.int64, [None, None, None, 1], name='labels')
        is_training = tf.placeholder(tf.bool, [])

        conv1_1 = self.new_conv_layer(imgs, [3,3,3,64], use_relu=True, name='conv1_1')
        _activation_summary('conv1_1', conv1_1)
        conv1_2 = self.new_conv_layer(conv1_1, [3,3,64,64], use_relu=True, name='conv1_2')
        _activation_summary('conv1_2', conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
                               
        conv2_1 = self.new_conv_layer(pool1, [3,3,64,128], use_relu=True, name='conv2_1')
        _activation_summary('conv2_1', conv2_1)
        conv2_2 = self.new_conv_layer(conv2_1, [3,3,128,128], use_relu=True, name='conv2_2')
        _activation_summary('conv2_2', conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')                               
                               
        conv3_1 = self.new_conv_layer(pool2, [3,3,128,256], use_relu=True, name='conv3_1')
        _activation_summary('conv3_1', conv3_1)
        conv3_2 = self.new_conv_layer(conv3_1, [3,3,256,256], use_relu=True, name='conv3_2')
        _activation_summary('conv3_2', conv3_2)
        conv3_3 = self.new_conv_layer(conv3_2, [3,3,256,256], use_relu=True, name='conv3_3')
        _activation_summary('conv3_3', conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
                               
        conv4_1 = self.new_conv_layer(pool3, [3,3,256,512], use_relu=True, name='conv4_1')
        _activation_summary('conv4_1', conv4_1)
        conv4_2 = self.new_conv_layer(conv4_1, [3,3,512,512], use_relu=True, name='conv4_2')
        _activation_summary('conv4_2', conv4_2)
        conv4_3 = self.new_conv_layer(conv4_2, [3,3,512,512], use_relu=True, name='conv4_3')
        _activation_summary('conv4_3', conv4_3)
        pool4= tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        
        conv5_1 = self.new_conv_layer(pool4, [3,3,512,512], use_relu=True, name='conv5_1')
        _activation_summary('conv5_1', conv5_1)
        conv5_2 = self.new_conv_layer(conv5_1, [3,3,512,512], use_relu=True, name='conv5_2')
        _activation_summary('conv5_2', conv5_2)
        conv5_3 = self.new_conv_layer(conv5_2, [3,3,512,512], use_relu=True, name='conv5_3')
        _activation_summary('conv5_3', conv5_3)

        # specialized layers
        target_shape = tf.concat(values=[tf.slice(tf.shape(imgs), [0], [3]),tf.constant(16,shape=[1,])], axis=0)  
        spe_1 = self.new_conv_layer(conv1_2, [3,3,64,16], use_relu=True, name='spe_1')
        _activation_summary('spe_1', spe_1)
        spe_2 = self.new_conv_layer(conv2_2, [3,3,128,16], use_relu=True, name='spe_2')
        _activation_summary('spe_2', spe_2)
        resized_spe_2 = self.new_deconv_layer(spe_2, [4,4,16,16], target_shape, [1,2,2,1], use_relu=True, name='resized_spe_2') 
        spe_3 = self.new_conv_layer(conv3_3, [3,3,256,16], use_relu=True, name='spe_3')
        _activation_summary('spe_3', spe_3)
        resized_spe_3 = self.new_deconv_layer(spe_3, [8,8,16,16], target_shape, [1,4,4,1], use_relu=True, name='resized_spe_3')
        spe_4 = self.new_conv_layer(conv4_3, [3,3,512,16], use_relu=True, name='spe_4')
        _activation_summary('spe_4', spe_4)
        resized_spe_4 = self.new_deconv_layer(spe_4, [16,16,16,16], target_shape, [1,8,8,1], use_relu=True, name='resized_spe_4')
        spe_5 = self.new_conv_layer(conv5_3, [3,3,512,16], use_relu=True, name='spe_5')
        _activation_summary('spe_5', spe_5)
        resized_spe_5 = self.new_deconv_layer(spe_5, [32,32,16,16], target_shape, [1,16,16,1], use_relu=True, name='resized_spe_5')
        spe_concat = tf.concat(values=[spe_1,resized_spe_2,resized_spe_3,resized_spe_4,resized_spe_5], axis=3)

        output = self.new_conv_layer(spe_concat, [1,1,16*5,1], name='output')
        _activation_summary('output', output)
        
        fg_prob = tf.sigmoid(output)

        ### Compute the loss ###
        binary_mask_fg = tf.to_float(tf.equal(labels, 1))
        binary_mask_bg = tf.to_float(tf.not_equal(labels, 1))
        combined_mask = tf.concat(values=[binary_mask_bg,binary_mask_fg], axis=3)
        flat_one_hot_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))
        flat_labels = tf.reshape(tensor=labels, shape=(-1,))
        flat_logits = tf.reshape(tensor=output, shape=(-1,))        
        cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=tf.to_float(flat_labels))
        
        """# simple cross entropy loss  
        loss = tf.reduce_mean(cross_entropies)"""
        # weighted cross entropy loss
        num_pixel = tf.size(labels)
        num_pixel_fg = tf.count_nonzero(binary_mask_fg, dtype=tf.int32)
        num_pixel_bg = num_pixel - num_pixel_fg
        class_weight = tf.cast(tf.concat(values=[tf.reshape(tf.divide(num_pixel_fg,num_pixel),(1,1)), \
                                                tf.reshape(tf.divide(num_pixel_bg,num_pixel),(1,1))], axis=1), dtype=tf.float32)
        weight_per_label = tf.transpose(tf.matmul(flat_one_hot_labels,tf.transpose(class_weight))) #shape [1, TRAIN.BATCH_SIZE]
        # this is the weight for each datapoint, depending on its label
        loss = tf.reduce_mean(tf.multiply(weight_per_label,cross_entropies))
         
        weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * cfg.TRAIN.WEIGHT_DECAY_RATE
        loss += weight_decay
        
        ### Compute the accuracy ###
        flat_bin_output = tf.greater_equal(tf.reshape(tensor=fg_prob, shape=(-1,)), 0.5)
        # accuracy
        correct = tf.to_float(tf.equal(flat_bin_output,tf.cast(flat_labels, tf.bool)))
        accuracy = tf.reduce_mean(correct)
        # precision, recall
        num_fg_output = tf.reduce_sum(tf.to_float(flat_bin_output))
        tp = tf.reduce_sum(tf.to_float(tf.logical_and(tf.cast(flat_labels, dtype=tf.bool), flat_bin_output)))
        pre = tf.divide(tp,tf.add(num_fg_output,cfg.EPSILON))
        rec = tf.divide(tp,tf.to_float(num_pixel_fg))
        
        ### Build the solver ###
        if self.params.opt=='adam':
            train_op = tf.train.AdamOptimizer(self.params.lr, epsilon=0.1).minimize(loss, global_step=self.global_step)
        elif self.params.opt=='sgd':
            if self.params.lr_decay=='const':
                # constant
                optimizer = tf.train.MomentumOptimizer(self.params.lr, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                grads_and_vars = optimizer.compute_gradients(loss)
                grads_and_vars = map(lambda gv: (0.01*gv[0],gv[1]) if 'output' in gv[1].name else (gv[0],gv[1]), grads_and_vars)
                grads_and_vars = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), grads_and_vars)
                #grads_and_vars = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), grads_and_vars)
                #grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
            elif self.params.lr_decay=='pc':
                # piecewise_constant
                boundaries = [int(self.params.max_iters*0.5),int(self.params.max_iters*0.75)]
                values = [self.params.lr,self.params.lr*0.5,self.params.lr*0.25]

                learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)
                #train_op = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True).minimize(loss, global_step=self.global_step)
                optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                grads_and_vars = optimizer.compute_gradients(loss)
                grads_and_vars = map(lambda gv: (0.01*gv[0],gv[1]) if 'output' in gv[1].name else (gv[0],gv[1]), grads_and_vars)
                grads_and_vars = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), grads_and_vars)
                #grads_and_vars = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), grads_and_vars)
                #grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
            elif self.params.lr_decay=='exp':
                # exponential_decay
                learning_rate = tf.train.exponential_decay(self.params.lr, self.global_step, self.params.max_iters/20, 0.9, staircase=False)
                #train_op = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True).minimize(loss, global_step=self.global_step)   
                optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                grads_and_vars = optimizer.compute_gradients(loss)
                grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        ### Hang up the results ###
        self.imgs = imgs
        self.labels = labels
        self.is_training = is_training
        
        self.conv_feats = spe_concat
        self.output = output
        self.fg_prob = fg_prob
        
        self.flat_bin_output = flat_bin_output
        self.tp = tp
        self.num_fg_output = num_fg_output
        self.num_pixel_fg = num_pixel_fg        
        
        self.accuracy = accuracy
        self.precision = pre        
        self.recall = rec
        
        self.loss = loss
        self.train_op = train_op

        
class vessel_segm_vgn(base_model):
    def __init__(self, params, weight_file_path):
        base_model.__init__(self, weight_file_path)
        self.params = params
        
        # cnn module related
        self.cnn_model = params.cnn_model
        self.cnn_loss_on = params.cnn_loss_on
        
        # gnn module related
        self.win_size = params.win_size
        self.gnn_loss_on = params.gnn_loss_on
        self.gnn_loss_weight = params.gnn_loss_weight
        
        # inference module related
        self.infer_module_kernel_size = params.infer_module_kernel_size

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.build_model()
        
    # special layer for graph attention network
    def sp_attn_head(self, bottom, output_size, adj, name, act=tf.nn.elu, feat_dropout=0., att_dropout=0., residual=False, show_adj=False):
       
        with tf.variable_scope(name) as scope:
            if feat_dropout != 0.0:
                bottom = tf.nn.dropout(bottom, 1.0 - feat_dropout)
    
            fts = tf.layers.conv1d(bottom, output_size, 1, use_bias=False)
    
            # simplest self-attention possible
            f_1 = tf.layers.conv1d(fts, 1, 1)
            f_2 = tf.layers.conv1d(fts, 1, 1)
            
            num_nodes = tf.slice(tf.shape(adj),[0],[1])
            f_1 = tf.reshape(f_1, tf.concat(values=[num_nodes,tf.constant(1,shape=[1,])], axis=0))
            f_2 = tf.reshape(f_2, tf.concat(values=[num_nodes,tf.constant(1,shape=[1,])], axis=0)) 
    
            f_1 = adj * f_1
            f_2 = adj * tf.transpose(f_2, [1,0])
    
            logits = tf.sparse_add(f_1, f_2)
            lrelu = tf.SparseTensor(indices=logits.indices, 
                    values=tf.nn.leaky_relu(logits.values), 
                    dense_shape=logits.dense_shape)
            coefs = tf.sparse_softmax(lrelu)
    
            if att_dropout != 0.0:
                coefs = tf.SparseTensor(indices=coefs.indices,
                        values=tf.nn.dropout(coefs.values, 1.0 - att_dropout),
                        dense_shape=coefs.dense_shape)
            if feat_dropout != 0.0:
                fts = tf.nn.dropout(fts, 1.0 - feat_dropout)
    
            coefs = tf.sparse_reshape(coefs, tf.concat(values=[num_nodes,num_nodes], axis=0))
            fts = tf.squeeze(fts, [0])
            vals = tf.sparse_tensor_dense_matmul(coefs, fts)
            vals = tf.expand_dims(vals, axis=0)
            vals = tf.reshape(vals, tf.concat(values=[tf.constant(1,shape=[1,]),num_nodes,tf.constant(output_size,shape=[1,])], axis=0))
            ret = tf.contrib.layers.bias_add(vals)
    
            # residual connection
            if residual:
                if bottom.shape[-1] != ret.shape[-1]:
                    ret = ret + tf.layers.conv1d(bottom, ret.shape[-1], 1) # activation
                else:
                    ret = ret + bottom
    
        if show_adj:
            #return act(ret), tf.sparse_reshape(lrelu, tf.concat(values=[num_nodes,num_nodes], axis=0))
            return act(ret), coefs
        else:
            return act(ret)
    
    def build_model(self):
        """Build the model."""
        print("Building the model...")
        self.build_cnn_module()
        self.build_gat() # GAT for our GNN module
        self.build_infer_module()
        self.build_optimizer()
        print("Model built.")
        
    def build_cnn_module(self):
        """Build the CNN module."""
        print("Building the CNN module...")
        if self.cnn_model=='driu':
            self.build_driu()
        elif self.cnn_model=='driu_large':
            self.build_driu_large()
        else:
            raise NotImplementedError
        print("CNN module built.")
        
    def build_driu(self):

        imgs = tf.placeholder(tf.float32, [None, None, None, 3], name='imgs') # note that the input is RGB
        labels = tf.placeholder(tf.int64, [None, None, None, 1], name='labels')
        fov_masks = tf.placeholder(tf.int64, [None, None, None, 1], name='fov_masks')

        conv1_1 = self.new_conv_layer(imgs, [3,3,3,64], use_relu=True, name='conv1_1')
        _activation_summary('conv1_1', conv1_1)
        conv1_2 = self.new_conv_layer(conv1_1, [3,3,64,64], use_relu=True, name='conv1_2')
        _activation_summary('conv1_2', conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
                               
        conv2_1 = self.new_conv_layer(pool1, [3,3,64,128], use_relu=True, name='conv2_1')
        _activation_summary('conv2_1', conv2_1)
        conv2_2 = self.new_conv_layer(conv2_1, [3,3,128,128], use_relu=True, name='conv2_2')
        _activation_summary('conv2_2', conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')                               
                               
        conv3_1 = self.new_conv_layer(pool2, [3,3,128,256], use_relu=True, name='conv3_1')
        _activation_summary('conv3_1', conv3_1)
        conv3_2 = self.new_conv_layer(conv3_1, [3,3,256,256], use_relu=True, name='conv3_2')
        _activation_summary('conv3_2', conv3_2)
        conv3_3 = self.new_conv_layer(conv3_2, [3,3,256,256], use_relu=True, name='conv3_3')
        _activation_summary('conv3_3', conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
                               
        conv4_1 = self.new_conv_layer(pool3, [3,3,256,512], use_relu=True, name='conv4_1')
        _activation_summary('conv4_1', conv4_1)
        conv4_2 = self.new_conv_layer(conv4_1, [3,3,512,512], use_relu=True, name='conv4_2')
        _activation_summary('conv4_2', conv4_2)
        conv4_3 = self.new_conv_layer(conv4_2, [3,3,512,512], use_relu=True, name='conv4_3')
        _activation_summary('conv4_3', conv4_3)

        # specialized layers
        target_shape = tf.concat(values=[tf.slice(tf.shape(imgs), [0], [3]),tf.constant(16,shape=[1,])], axis=0)  
        spe_1 = self.new_conv_layer(conv1_2, [3,3,64,16], use_relu=True, name='spe_1')
        _activation_summary('spe_1', spe_1)
        spe_2 = self.new_conv_layer(conv2_2, [3,3,128,16], use_relu=True, name='spe_2')
        _activation_summary('spe_2', spe_2)
        resized_spe_2 = self.new_deconv_layer(spe_2, [4,4,16,16], target_shape, [1,2,2,1], use_relu=True, name='resized_spe_2') 
        spe_3 = self.new_conv_layer(conv3_3, [3,3,256,16], use_relu=True, name='spe_3')
        _activation_summary('spe_3', spe_3)
        resized_spe_3 = self.new_deconv_layer(spe_3, [8,8,16,16], target_shape, [1,4,4,1], use_relu=True, name='resized_spe_3')
        spe_4 = self.new_conv_layer(conv4_3, [3,3,512,16], use_relu=True, name='spe_4')
        _activation_summary('spe_4', spe_4)
        resized_spe_4 = self.new_deconv_layer(spe_4, [16,16,16,16], target_shape, [1,8,8,1], use_relu=True, name='resized_spe_4')  
        spe_concat = tf.concat(values=[spe_1,resized_spe_2,resized_spe_3,resized_spe_4], axis=3)
        
        img_output = self.new_conv_layer(spe_concat, [1,1,16*4,1], name='img_output')
        _activation_summary('img_output', img_output)
        
        img_fg_prob = tf.sigmoid(img_output)
        
        ### Hang up the results ###
        self.imgs = imgs
        self.labels = labels
        self.fov_masks = fov_masks
        
        self.cnn_feat = {}
        self.cnn_feat[1] = spe_1
        self.cnn_feat[2] = spe_2
        self.cnn_feat[4] = spe_3
        self.cnn_feat[8] = spe_4
        
        self.cnn_feat_spatial_sizes = {}
        self.cnn_feat_spatial_sizes[1] = tf.slice(tf.shape(spe_1),[1],[2])
        self.cnn_feat_spatial_sizes[2] = tf.slice(tf.shape(spe_2),[1],[2])
        self.cnn_feat_spatial_sizes[4] = tf.slice(tf.shape(spe_3),[1],[2])
        self.cnn_feat_spatial_sizes[8] = tf.slice(tf.shape(spe_4),[1],[2])
        
        self.conv_feats = spe_concat
        self.img_output = img_output
        self.img_fg_prob = img_fg_prob
        
        self.var_to_restore = ['conv1_1','conv1_2', \
                               'conv2_1','conv2_2', \
                               'conv3_1','conv3_2','conv3_3', \
                               'conv4_1','conv4_2','conv4_3', \
                               'spe_1', 'spe_2', 'spe_3', 'spe_4', \
                               'resized_spe_2', 'resized_spe_3', 'resized_spe_4', \
                               'img_output']
        
    def build_driu_large(self):

        imgs = tf.placeholder(tf.float32, [None, None, None, 3], name='imgs') # note that the input is RGB
        labels = tf.placeholder(tf.int64, [None, None, None, 1], name='labels')
        fov_masks = tf.placeholder(tf.int64, [None, None, None, 1], name='fov_masks')

        conv1_1 = self.new_conv_layer(imgs, [3,3,3,64], use_relu=True, name='conv1_1')
        _activation_summary('conv1_1', conv1_1)
        conv1_2 = self.new_conv_layer(conv1_1, [3,3,64,64], use_relu=True, name='conv1_2')
        _activation_summary('conv1_2', conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
                               
        conv2_1 = self.new_conv_layer(pool1, [3,3,64,128], use_relu=True, name='conv2_1')
        _activation_summary('conv2_1', conv2_1)
        conv2_2 = self.new_conv_layer(conv2_1, [3,3,128,128], use_relu=True, name='conv2_2')
        _activation_summary('conv2_2', conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')                               
                               
        conv3_1 = self.new_conv_layer(pool2, [3,3,128,256], use_relu=True, name='conv3_1')
        _activation_summary('conv3_1', conv3_1)
        conv3_2 = self.new_conv_layer(conv3_1, [3,3,256,256], use_relu=True, name='conv3_2')
        _activation_summary('conv3_2', conv3_2)
        conv3_3 = self.new_conv_layer(conv3_2, [3,3,256,256], use_relu=True, name='conv3_3')
        _activation_summary('conv3_3', conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
                               
        conv4_1 = self.new_conv_layer(pool3, [3,3,256,512], use_relu=True, name='conv4_1')
        _activation_summary('conv4_1', conv4_1)
        conv4_2 = self.new_conv_layer(conv4_1, [3,3,512,512], use_relu=True, name='conv4_2')
        _activation_summary('conv4_2', conv4_2)
        conv4_3 = self.new_conv_layer(conv4_2, [3,3,512,512], use_relu=True, name='conv4_3')
        _activation_summary('conv4_3', conv4_3)
        pool4= tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name='pool4')
        
        conv5_1 = self.new_conv_layer(pool4, [3,3,512,512], use_relu=True, name='conv5_1')
        _activation_summary('conv5_1', conv5_1)
        conv5_2 = self.new_conv_layer(conv5_1, [3,3,512,512], use_relu=True, name='conv5_2')
        _activation_summary('conv5_2', conv5_2)
        conv5_3 = self.new_conv_layer(conv5_2, [3,3,512,512], use_relu=True, name='conv5_3')
        _activation_summary('conv5_3', conv5_3)

        # specialized layers
        target_shape = tf.concat(values=[tf.slice(tf.shape(imgs), [0], [3]),tf.constant(16,shape=[1,])], axis=0)  
        spe_1 = self.new_conv_layer(conv1_2, [3,3,64,16], use_relu=True, name='spe_1')
        _activation_summary('spe_1', spe_1)
        spe_2 = self.new_conv_layer(conv2_2, [3,3,128,16], use_relu=True, name='spe_2')
        _activation_summary('spe_2', spe_2)
        resized_spe_2 = self.new_deconv_layer(spe_2, [4,4,16,16], target_shape, [1,2,2,1], use_relu=True, name='resized_spe_2') 
        spe_3 = self.new_conv_layer(conv3_3, [3,3,256,16], use_relu=True, name='spe_3')
        _activation_summary('spe_3', spe_3)
        resized_spe_3 = self.new_deconv_layer(spe_3, [8,8,16,16], target_shape, [1,4,4,1], use_relu=True, name='resized_spe_3')
        spe_4 = self.new_conv_layer(conv4_3, [3,3,512,16], use_relu=True, name='spe_4')
        _activation_summary('spe_4', spe_4)
        resized_spe_4 = self.new_deconv_layer(spe_4, [16,16,16,16], target_shape, [1,8,8,1], use_relu=True, name='resized_spe_4')  
        spe_5 = self.new_conv_layer(conv5_3, [3,3,512,16], use_relu=True, name='spe_5')
        _activation_summary('spe_5', spe_5)
        resized_spe_5 = self.new_deconv_layer(spe_5, [32,32,16,16], target_shape, [1,16,16,1], use_relu=True, name='resized_spe_5')
        spe_concat = tf.concat(values=[spe_1,resized_spe_2,resized_spe_3,resized_spe_4,resized_spe_5], axis=3)
        
        img_output = self.new_conv_layer(spe_concat, [1,1,16*5,1], name='img_output')
        _activation_summary('img_output', img_output)
        
        img_fg_prob = tf.sigmoid(img_output)
        
        ### Hang up the results ###
        self.imgs = imgs
        self.labels = labels
        self.fov_masks = fov_masks
        
        self.cnn_feat = {}
        self.cnn_feat[1] = spe_1
        self.cnn_feat[2] = spe_2
        self.cnn_feat[4] = spe_3
        self.cnn_feat[8] = spe_4
        self.cnn_feat[16] = spe_5
        
        self.cnn_feat_spatial_sizes = {}
        self.cnn_feat_spatial_sizes[1] = tf.slice(tf.shape(spe_1),[1],[2])
        self.cnn_feat_spatial_sizes[2] = tf.slice(tf.shape(spe_2),[1],[2])
        self.cnn_feat_spatial_sizes[4] = tf.slice(tf.shape(spe_3),[1],[2])
        self.cnn_feat_spatial_sizes[8] = tf.slice(tf.shape(spe_4),[1],[2])
        self.cnn_feat_spatial_sizes[16] = tf.slice(tf.shape(spe_5),[1],[2])
        
        self.conv_feats = spe_concat
        self.img_output = img_output
        self.img_fg_prob = img_fg_prob
        
        self.var_to_restore = ['conv1_1','conv1_2', \
                               'conv2_1','conv2_2', \
                               'conv3_1','conv3_2','conv3_3', \
                               'conv4_1','conv4_2','conv4_3', \
                               'conv5_1','conv5_2','conv5_3', \
                               'spe_1', 'spe_2', 'spe_3', 'spe_4', 'spe_5', \
                               'resized_spe_2', 'resized_spe_3', 'resized_spe_4', 'resized_spe_5', \
                               'img_output']
        
    def build_gat(self):
        """Build the GAT."""
        print("Building the GAT part...")
        
        node_byxs = tf.placeholder(tf.int32, [None, 3], name='node_byxs')
        adj = tf.sparse_placeholder(tf.float32, [None, None], name='adj')
        node_feats = tf.gather_nd(self.conv_feats, node_byxs, name='node_feats')
        node_labels = tf.cast(tf.reshape(tf.gather_nd(self.labels, node_byxs), [-1]), tf.float32, name='node_labels')
        
        node_feats_resh = tf.expand_dims(node_feats, axis=0)

        gnn_feat_dropout = tf.placeholder_with_default(0., shape=())
        gnn_att_dropout = tf.placeholder_with_default(0., shape=())
        
        layer_name_list = []
        attns = []
        for head_idx in range(self.params.gat_n_heads[0]):
            cur_name = 'gat_hidden_1_%d'%(head_idx+1)
            layer_name_list.append(cur_name)
            attns.append(self.sp_attn_head(node_feats_resh, self.params.gat_hid_units[0], adj, \
                                           name=cur_name, \
                                           feat_dropout=gnn_feat_dropout, att_dropout=gnn_att_dropout, \
                                           residual=self.params.gat_use_residual))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(self.params.gat_hid_units)):
            attns = []
            for head_idx in range(self.params.gat_n_heads[i]):
                cur_name = 'gat_hidden_%d_%d'%(i+1,head_idx+1)
                layer_name_list.append(cur_name)
                attns.append(self.sp_attn_head(h_1, self.params.gat_hid_units[i], adj, \
                                               name=cur_name, \
                                               feat_dropout=gnn_feat_dropout, att_dropout=gnn_att_dropout, \
                                               residual=self.params.gat_use_residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for head_idx in range(self.params.gat_n_heads[-1]):
            cur_name = 'gat_node_logits_%d'%(head_idx+1)
            layer_name_list.append(cur_name)
            out.append(self.sp_attn_head(h_1, 1, adj, \
                                             name=cur_name, \
                                             act=lambda x: x, \
                                             feat_dropout=gnn_feat_dropout, att_dropout=gnn_att_dropout, \
                                             residual=False)) 
        
        node_logits = tf.add_n(out) / self.params.gat_n_heads[-1]
        node_logits = tf.squeeze(node_logits, [0,2])
        
        ### Hang up the results ###
        self.node_logits = node_logits # [num_nodes,]
        self.gnn_final_feats = tf.squeeze(h_1) # [num_nodes, self.params.gat_hid_units[-1]]
                
        self.node_byxs = node_byxs
        self.node_feats = node_feats
        self.node_labels = node_labels
        self.adj = adj
        
        self.gnn_feat_dropout = gnn_feat_dropout
        self.gnn_att_dropout = gnn_att_dropout
        
        self.var_to_restore += layer_name_list
        
        print("GAT part built.")
        
    def build_infer_module(self):
        """Build the inference module."""
        print("Building the inference module...")
        # please note that all layers/variables related to the inference module
        # are prefixed with 'post_cnn' instead of 'infer_module'

        post_cnn_dropout = tf.placeholder_with_default(0., shape=())
        
        is_lr_flipped = tf.placeholder(tf.bool, [])
        is_ud_flipped = tf.placeholder(tf.bool, [])
        rot90_num = tf.placeholder_with_default(0., shape=())
        
        y_len = tf.cast(tf.ceil(tf.divide(tf.to_float(tf.slice(tf.shape(self.imgs),[1],[1])),self.win_size)), dtype=tf.int32)
        x_len = tf.cast(tf.ceil(tf.divide(tf.to_float(tf.slice(tf.shape(self.imgs),[2],[1])),self.win_size)), dtype=tf.int32)
        
        sp_size = tf.cond(tf.logical_or(tf.equal(rot90_num,0), tf.equal(rot90_num,2)), \
                          lambda: tf.concat(values=[y_len,x_len], axis=0), \
                          lambda: tf.concat(values=[x_len,y_len], axis=0))
            
        reshaped_gnn_feats = tf.reshape(tensor=self.gnn_final_feats, \
                                        shape=tf.concat(values=[tf.slice(tf.shape(self.imgs),[0],[1]), \
                                                                sp_size, \
                                                                tf.slice(tf.shape(self.gnn_final_feats),[1],[1])], axis=0))
        
        reshaped_gnn_feats = tf.cond(is_lr_flipped, \
            lambda: tf.image.flip_left_right(reshaped_gnn_feats) , \
            lambda: reshaped_gnn_feats)
        
        reshaped_gnn_feats = tf.cond(is_ud_flipped, \
            lambda: tf.image.flip_up_down(reshaped_gnn_feats) , \
            lambda: reshaped_gnn_feats)
        
        reshaped_gnn_feats = tf.cond(tf.math.not_equal(rot90_num,0), \
            lambda: tf.image.rot90(reshaped_gnn_feats, tf.cast(rot90_num, tf.int32)), \
            lambda: reshaped_gnn_feats)
                                                    
        temp_num_chs = self.params.gat_n_heads[-2]*self.params.gat_hid_units[-1]

        post_cnn_conv_comp = self.new_conv_layer(reshaped_gnn_feats, [1,1,temp_num_chs,32], norm_type=self.params.norm_type, use_relu=True, name='post_cnn_conv_comp') # changed
        current_input = post_cnn_conv_comp
        ds_rate = self.win_size/2
        while ds_rate>=1:
            
            cur_deconv_name = 'post_cnn_deconv%d'%(ds_rate)
            upsampled = self.new_deconv_layer(current_input, [4,4,16,32], \
                                              tf.concat(values=[tf.slice(tf.shape(self.imgs),[0],[1]), \
                                                                self.cnn_feat_spatial_sizes[ds_rate], \
                                                                tf.constant(16,shape=[1,])], axis=0), \
                                              [1,2,2,1], norm_type=self.params.norm_type, use_relu=True, name=cur_deconv_name)
            
            cur_cnn_feat = tf.nn.dropout(self.cnn_feat[ds_rate], 1-post_cnn_dropout)
            if self.params.use_enc_layer: 
                cur_cnn_feat = self.new_conv_layer(cur_cnn_feat, [1,1,16,16], \
                                                   norm_type=self.params.norm_type, use_relu=True, name='post_cnn_cnn_feat%d'%(ds_rate)) # added
            else:
                if self.params.norm_type=='BN':
                    #cur_cnn_feat = tf.layers.batch_normalization(cur_cnn_feat, training=is_training, renorm=cfg.USE_BRN)
                    pass
                elif self.params.norm_type=='GN':
                    cur_cnn_feat = self.group_norm_layer(cur_cnn_feat, num_group=min(cfg.GN_MIN_NUM_G, 16/cfg.GN_MIN_CHS_PER_G), name='post_cnn_cnn_feat%d'%(ds_rate))
                    cur_cnn_feat = tf.nn.relu(cur_cnn_feat)
                else:
                    pass

            if ds_rate==1:
                cur_conv_name = 'post_cnn_img_output'
                output = self.new_conv_layer(tf.concat(values=[upsampled,cur_cnn_feat], axis=3), \
                                             [self.infer_module_kernel_size,self.infer_module_kernel_size,32,1], \
                                             name=cur_conv_name) # changed
                self.post_cnn_img_output = output
            else:
                cur_conv_name = 'post_cnn_conv%d'%(ds_rate)
                output = self.new_conv_layer(tf.concat(values=[upsampled,cur_cnn_feat], axis=3), \
                                             [self.infer_module_kernel_size,self.infer_module_kernel_size,32,32], \
                                             norm_type=self.params.norm_type, use_relu=True, name=cur_conv_name) # changed
            
            current_input = output
            ds_rate = ds_rate/2

        post_cnn_img_fg_prob = tf.sigmoid(current_input)
        
        pixel_weights = tf.placeholder(tf.float32, [None, None, None, 1], name='pixel_weights')
        
        ### Hang up the results ###     
        self.post_cnn_dropout = post_cnn_dropout
        
        self.pixel_weights = pixel_weights
        self.post_cnn_img_fg_prob = post_cnn_img_fg_prob
        
        self.is_lr_flipped = is_lr_flipped
        self.is_ud_flipped = is_ud_flipped
        self.rot90_num = rot90_num
        self.reshaped_gnn_feats = reshaped_gnn_feats
        
        print("inference module built.")
        
    def build_optimizer(self):
        """Build the optimizer."""
        print("Building the optimizer part...")
        
        ###### cnn related ######
        
        ### Compute the loss ###
        binary_mask_fg = tf.to_float(tf.equal(self.labels, 1))
        binary_mask_bg = tf.to_float(tf.not_equal(self.labels, 1))
        combined_mask = tf.concat(values=[binary_mask_bg,binary_mask_fg], axis=3)
        flat_one_hot_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))
        flat_labels = tf.reshape(tensor=self.labels, shape=(-1,))
        flat_logits = tf.reshape(tensor=self.img_output, shape=(-1,))
        cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=tf.to_float(flat_labels))
        
        """# weighted cross entropy loss (in/out fov)
        num_pixel = tf.size(self.labels)
        num_pixel_fg = tf.count_nonzero(binary_mask_fg, dtype=tf.int32)
        num_pixel_bg = num_pixel - num_pixel_fg
        class_weight = tf.cast(tf.concat(values=[tf.reshape(tf.divide(num_pixel_fg,num_pixel),(1,1)), \
                                                tf.reshape(tf.divide(num_pixel_bg,num_pixel),(1,1))], axis=1), dtype=tf.float32)
        weight_per_label = tf.transpose(tf.matmul(flat_one_hot_labels,tf.transpose(class_weight))) #shape [1, TRAIN.BATCH_SIZE]
        # this is the weight for each datapoint, depending on its label
        cnn_loss = tf.reduce_mean(tf.multiply(weight_per_label,cross_entropies))"""
        
        # weighted cross entropy loss (in fov)
        num_pixel = tf.reduce_sum(self.fov_masks)
        num_pixel_fg = tf.count_nonzero(binary_mask_fg, dtype=tf.int64)
        num_pixel_bg = num_pixel - num_pixel_fg
        class_weight = tf.cast(tf.concat(values=[tf.reshape(tf.divide(num_pixel_fg,num_pixel),(1,1)), \
                                                tf.reshape(tf.divide(num_pixel_bg,num_pixel),(1,1))], axis=1), dtype=tf.float32)
        weight_per_label = tf.transpose(tf.matmul(flat_one_hot_labels,tf.transpose(class_weight))) #shape [1, TRAIN.BATCH_SIZE]
        # this is the weight for each datapoint, depending on its label
        reshaped_fov_masks = tf.reshape(tensor=tf.cast(self.fov_masks, tf.float32), shape=(-1,))
        reshaped_fov_masks /= tf.reduce_mean(reshaped_fov_masks)
        cnn_loss = tf.reduce_mean(tf.multiply(tf.multiply(reshaped_fov_masks, weight_per_label), cross_entropies))
        
        ### Compute the accuracy ###
        flat_bin_output = tf.greater_equal(tf.reshape(tensor=self.img_fg_prob, shape=(-1,)), 0.5)
        # accuracy
        cnn_correct = tf.to_float(tf.equal(flat_bin_output,tf.cast(flat_labels, tf.bool)))
        cnn_accuracy = tf.reduce_mean(cnn_correct)
        # precision, recall
        num_fg_output = tf.reduce_sum(tf.to_float(flat_bin_output)) 
        cnn_tp = tf.reduce_sum(tf.to_float(tf.logical_and(tf.cast(flat_labels, dtype=tf.bool), flat_bin_output)))
        cnn_pre = tf.divide(cnn_tp,tf.add(num_fg_output,cfg.EPSILON))
        cnn_rec = tf.divide(cnn_tp,tf.to_float(num_pixel_fg))


        ###### gnn related ######

        ### Compute the loss ###
        gnn_cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.node_logits, labels=self.node_labels)
        
        # simple cross entropy loss
        #gnn_loss = tf.reduce_mean(gnn_cross_entropies)
        # weighted cross entropy loss
        num_node = tf.size(self.node_labels)
        num_node_fg = tf.count_nonzero(self.node_labels, dtype=tf.int32)
        num_node_bg = num_node - num_node_fg
        gnn_class_weight = tf.cast(tf.concat(values=[tf.reshape(tf.divide(num_node_fg,num_node),(1,1)), \
                                                     tf.reshape(tf.divide(num_node_bg,num_node),(1,1))], axis=1), dtype=tf.float32)
        gnn_weight_per_label = tf.transpose(tf.matmul(tf.one_hot(tf.cast(self.node_labels, tf.int32), 2), tf.transpose(gnn_class_weight))) #shape [1, TRAIN.BATCH_SIZE]
        # this is the weight for each datapoint, depending on its label
        gnn_loss = tf.reduce_mean(tf.multiply(gnn_weight_per_label,gnn_cross_entropies))

        ### Compute the accuracy ###
        gnn_prob = tf.sigmoid(self.node_logits)
        gnn_correct = tf.equal(tf.cast(tf.greater_equal(gnn_prob, 0.5), tf.int32), tf.cast(self.node_labels, tf.int32))
        gnn_accuracy = tf.reduce_mean(tf.cast(gnn_correct, tf.float32))
        
        
        ###### inference module related ######
        
        ### Compute the loss ###
        post_cnn_flat_logits = tf.reshape(tensor=self.post_cnn_img_output, shape=(-1,))
        post_cnn_cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=post_cnn_flat_logits, labels=tf.to_float(flat_labels))
        
        # weighted cross entropy loss
        reshaped_pixel_weights = tf.reshape(tensor=self.pixel_weights, shape=(-1,))
        reshaped_pixel_weights /= tf.reduce_mean(reshaped_pixel_weights)
        post_cnn_loss = tf.reduce_mean(tf.multiply(tf.multiply(reshaped_pixel_weights, weight_per_label), post_cnn_cross_entropies))
        
        ### Compute the accuracy ###
        post_cnn_flat_bin_output = tf.greater_equal(tf.reshape(tensor=self.post_cnn_img_fg_prob, shape=(-1,)), 0.5)
        # accuracy
        post_cnn_correct = tf.to_float(tf.equal(post_cnn_flat_bin_output,tf.cast(flat_labels, tf.bool)))
        post_cnn_accuracy = tf.reduce_mean(post_cnn_correct)
        # precision, recall
        post_cnn_num_fg_output = tf.reduce_sum(tf.to_float(post_cnn_flat_bin_output))
        post_cnn_tp = tf.reduce_sum(tf.to_float(tf.logical_and(tf.cast(flat_labels, dtype=tf.bool), post_cnn_flat_bin_output)))
        post_cnn_pre = tf.divide(post_cnn_tp,tf.add(post_cnn_num_fg_output,cfg.EPSILON))
        post_cnn_rec = tf.divide(post_cnn_tp,tf.to_float(num_pixel_fg))

        ###### joint optimization ######
        
        if self.cnn_loss_on:
            loss = tf.add_n([cnn_loss, post_cnn_loss])
        else:
            loss = post_cnn_loss
        
        weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * cfg.TRAIN.WEIGHT_DECAY_RATE
        loss += weight_decay
        
        ### Build the solver ###
        learning_rate = tf.placeholder(tf.float32, shape=[])
        if self.params.opt=='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1)
        elif self.params.opt=='sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True)

        grads_and_vars_1 = optimizer.compute_gradients(loss)
        if self.gnn_loss_on:
            grads_and_vars_2 = optimizer.compute_gradients(gnn_loss*self.gnn_loss_weight)
            grads_and_vars_2 = map(lambda gv: (tf.clip_by_value(gv[0], -5., 5.),gv[1]) \
                                   if (('gat' in gv[1].name) and (gv[0] is not None)) \
                                   else (None,gv[1]), grads_and_vars_2)
            grads_and_vars = map(lambda gv_tuple: (add_tensors_wo_none([gv_tuple[0][0],gv_tuple[1][0]]), gv_tuple[0][1]), list(zip(grads_and_vars_1,grads_and_vars_2)))     
        else:
            grads_and_vars = grads_and_vars_1
            
        if self.params.old_net_ft_lr==0:
            # update only the newly added sub-network
            if self.params.lr_scheduling=='pc':
                boundaries = [int(self.params.max_iters*self.params.lr_decay_tp)]
                values = [self.params.new_net_lr,self.params.new_net_lr*0.1]            
                lr_handler = tf.train.piecewise_constant(self.global_step, boundaries, values)
            else:
                raise NotImplementedError
                
            if self.params.do_simul_training:
                grads_and_vars = map(lambda gv: (tf.clip_by_value(gv[0], -5., 5.),gv[1]) \
                                     if (('gat' in gv[1].name or 'post_cnn' in gv[1].name) and (gv[0] is not None)) \
                                     else (None,gv[1]), grads_and_vars)
            else:
                grads_and_vars = map(lambda gv: (tf.clip_by_value(gv[0], -5., 5.),gv[1]) if (('post_cnn' in gv[1].name) and (gv[0] is not None)) \
                                     else (None,gv[1]), grads_and_vars)
                
            grads_and_vars = map(lambda gv: (gv[0]*self.params.infer_module_grad_weight,gv[1]) \
                                 if 'post_cnn' in gv[1].name \
                                 else (gv[0],gv[1]), grads_and_vars)     
                
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)    
        else:
            # update the whole network
            if self.params.lr_scheduling=='pc':
                boundaries = [int(self.params.max_iters*self.params.lr_decay_tp)]
                values = [self.params.old_net_ft_lr,self.params.old_net_ft_lr*0.1]         
                lr_handler = tf.train.piecewise_constant(self.global_step, boundaries, values)
            else:
                raise NotImplementedError

            lr_ratio = self.params.new_net_lr/self.params.old_net_ft_lr
            if self.params.do_simul_training:
                grads_and_vars = map(lambda gv: (lr_ratio*gv[0],gv[1]) \
                                     if (('gat' in gv[1].name or 'post_cnn' in gv[1].name) and (gv[0] is not None)) \
                                     else (gv[0],gv[1]), grads_and_vars)
            else:
                grads_and_vars = map(lambda gv: (lr_ratio*gv[0],gv[1]) \
                                     if (('post_cnn' in gv[1].name) and (gv[0] is not None)) \
                                     else (gv[0],gv[1]), grads_and_vars)
            
            grads_and_vars = map(lambda gv: (tf.clip_by_value(gv[0], -5., 5.),gv[1]) \
                                 if gv[0] is not None \
                                 else (gv[0],gv[1]), grads_and_vars)
            
            grads_and_vars = map(lambda gv: (gv[0]*self.params.infer_module_grad_weight,gv[1]) \
                                 if 'post_cnn' in gv[1].name \
                                 else (gv[0],gv[1]), grads_and_vars)  
            
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        ### Hang up the results ###  
        self.cnn_flat_bin_output = flat_bin_output
        self.cnn_loss = cnn_loss
        self.cnn_tp = cnn_tp
        self.cnn_num_fg_output = num_fg_output
        self.cnn_num_pixel_fg = num_pixel_fg
        self.cnn_accuracy = cnn_accuracy
        self.cnn_precision = cnn_pre        
        self.cnn_recall = cnn_rec
        
        self.gnn_prob = gnn_prob
        self.gnn_loss = gnn_loss
        self.gnn_accuracy = gnn_accuracy
        
        self.post_cnn_flat_bin_output = post_cnn_flat_bin_output
        self.post_cnn_loss = post_cnn_loss
        self.post_cnn_tp = post_cnn_tp
        self.post_cnn_num_fg_output = post_cnn_num_fg_output
        self.post_cnn_accuracy = post_cnn_accuracy
        self.post_cnn_precision = post_cnn_pre        
        self.post_cnn_recall = post_cnn_rec

        self.learning_rate = learning_rate
        self.lr_handler = lr_handler

        self.loss = loss
        self.train_op = train_op

        print("optimizer part built.")