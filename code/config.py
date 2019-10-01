""" Common config file
"""


import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C



##### Training (general) #####

__C.TRAIN = edict()

__C.TRAIN.MODEL_SAVE_PATH = 'train'

__C.TRAIN.DISPLAY = 10

__C.TRAIN.TEST_ITERS = 500

__C.TRAIN.SNAPSHOT_ITERS = 500

__C.TRAIN.WEIGHT_DECAY_RATE = 0.0005

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.BATCH_SIZE = 1 # for CNN

__C.TRAIN.GRAPH_BATCH_SIZE = 1 # for VGN

##### Training (paths) #####

__C.TRAIN.DRIVE_SET_TXT_PATH = '../DRIVE/train.txt'

__C.TRAIN.STARE_SET_TXT_PATH = '../STARE/train.txt'

__C.TRAIN.CHASE_DB1_SET_TXT_PATH = '../CHASE_DB1/train.txt'

__C.TRAIN.HRF_SET_TXT_PATH = '../HRF/train_768.txt'

__C.TRAIN.TEMP_GRAPH_SAVE_PATH = 'graph'

##### Training (augmentation) #####

# horizontal flipping
__C.TRAIN.USE_LR_FLIPPED = True

# vertical flipping
__C.TRAIN.USE_UD_FLIPPED = False

# rotation
__C.TRAIN.USE_ROTATION = False
__C.TRAIN.ROTATION_MAX_ANGLE = 45

# scaling
__C.TRAIN.USE_SCALING = False
__C.TRAIN.SCALING_RANGE = [1., 1.25]

# cropping
__C.TRAIN.USE_CROPPING = False
__C.TRAIN.CROPPING_MAX_MARGIN = 0.05 # in ratio

# brightness adjustment
__C.TRAIN.USE_BRIGHTNESS_ADJUSTMENT = True
__C.TRAIN.BRIGHTNESS_ADJUSTMENT_MAX_DELTA = 0.2

# contrast adjustment
__C.TRAIN.USE_CONTRAST_ADJUSTMENT = True
__C.TRAIN.CONTRAST_ADJUSTMENT_LOWER_FACTOR = 0.5
__C.TRAIN.CONTRAST_ADJUSTMENT_UPPER_FACTOR = 1.5



##### Test (general) #####

__C.TEST = edict()

##### Test (paths) #####

__C.TEST.DRIVE_SET_TXT_PATH = '../DRIVE/test.txt'

__C.TEST.STARE_SET_TXT_PATH = '../STARE/test.txt'

__C.TEST.CHASE_DB1_SET_TXT_PATH = '../CHASE_DB1/test.txt'

__C.TEST.HRF_SET_TXT_PATH = '../HRF/test_768.txt'
#__C.TEST.HRF_SET_TXT_PATH = '../HRF/test_fr.txt'

__C.TEST.RES_SAVE_PATH = 'test'

# especially for the HRF dataset
__C.TEST.WHOLE_IMG_RES_SAVE_PATH = 'test_whole_img'



##### Misc. #####

__C.PIXEL_MEAN_DRIVE = [126.837, 69.015, 41.422]

__C.PIXEL_MEAN_STARE = [150.296, 83.550, 27.501]

__C.PIXEL_MEAN_CHASE_DB1 = [113.953, 39.807, 6.880]

__C.PIXEL_MEAN_HRF = [164.420, 51.826, 27.130]

__C.EPSILON = 1e-03



##### Feature normalization #####

__C.USE_BRN = True

__C.GN_MIN_NUM_G = 8

__C.GN_MIN_CHS_PER_G = 16