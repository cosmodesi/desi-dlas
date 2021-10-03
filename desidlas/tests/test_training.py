#This training process need to use GPU, users can download this file and run the test on their own computer.

import numpy as np
import math
import re, os, traceback, sys, json
import argparse
import tensorflow as tf
import os
from pathlib import Path

from tensorflow.python.framework import ops
ops.reset_default_graph()

from model import build_model

#Parameter configuration for tensorflow session
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True #Dynamically apply for video memory, slowly increase gpu capacity from less to more
config.allow_soft_placement=True  #Allow tensorflow to automatically allocate devices
config.log_device_placement = True #Record the log of which device each node is assigned to, for the convenience of debugging


tensor_regex = re.compile('.*:\d*')

from training import t
from training import train_ann_test_batch
from training import train_ann
from training import save_checkpoint
from training import calc_normalized_score

from parameterset import parameter_names
from parameterset import parameters
    
train_dataset='desidlas/tests/datafile/github_training_datasets.npy'
test_dataset='desidlas/tests/datafile/github_testing_datasets.npy'
checkpoint_filename='desidlas/tests/datafile/current'#path to save the model file

hyperparameters = {}
#choose the hyperparameters
for k in range(0,len(parameter_names)):
  hyperparameters[parameter_names[k]] = parameters[k][0]

remote_data = pytest.mark.skipif(os.getenv('CNN_MODEL') is None,
                                 reason='test requires GPU')

@remote_data

#start the training
#this process does not use CNN model files, but it need many GPU and may take a long time. So we still suggest the users run this test on their own pc.
def test_training():
    (best_accuracy, last_accuracy, last_objective, best_offset_rmse, last_offset_rmse, best_coldensity_rmse,
    last_coldensity_rmse) = train_ann(hyperparameters, train_dataset, test_dataset,
                                    save_filename=checkpoint_filename, load_filename= None)
    
