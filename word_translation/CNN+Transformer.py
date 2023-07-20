import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow.keras.mixed_precision as mixed_precision
from tqdm.autonotebook import tqdm
import sklearn
from tf_utils.schedules import OneCycleLR, ListedLR
from tf_utils.callbacks import Snapshot, SWA
from tf_utils.learners import FGM, AWP
import os
import time
import pickle
import math
import random
import sys
import cv2
import gc
import glob
import datetime

print(f'Tensorflow Version: {tf.__version__}')
print(f'Python Version: {sys.version}')


class CFG:
    n_splits = 5
    save_output = True
    output_dir = '/kaggle/working'

    seed = 42
    verbose = 2  # 0) silent 1) progress bar 2) one line per epoch

    max_len = 384
    replicas = 8
    lr = 5e-4 * replicas
    weight_decay = 0.1
    lr_min = 1e-6
    epoch = 300  # 400
    warmup = 0
    batch_size = 64 * replicas
    snapshot_epochs = []
    swa_epochs = []  # list(range(epoch//2,epoch+1))

    fp16 = True
    fgm = False
    awp = True
    awp_lambda = 0.2
    awp_start_epoch = 15
    dropout_start_epoch = 15
    resume = 0
    decay_type = 'cosine'
    dim = 192
    comment = f'islr-fp16-192-8-seed{seed}'

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




TRAIN_FILENAMES = glob.glob('/kaggle/input/islr-5fold/*.tfrecords')
print(len(TRAIN_FILENAMES))