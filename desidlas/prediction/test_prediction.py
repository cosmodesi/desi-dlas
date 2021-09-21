import numpy as np
import math
import re, os, traceback, sys, json
import argparse
import tensorflow as tf
import timeit
from tensorflow.python.framework import ops

ops.reset_default_graph()


from desidlas.training.model import build_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
tensor_regex = re.compile('.*:\d*')


from get_partprediction import t
from get_partprediction import predictions_ann

from desidlas.training.parameterset import parameter_names
from desidlas.training.parameterset import parameters
hyperparameters = {}
for k in range(0,len(parameter_names)):
  hyperparameters[parameter_names[k]] = parameters[k][0]
  
pred_dataset='testdata/'
savefile='testdata/'

r=np.load(pred_dataset,allow_pickle = True,encoding='latin1').item()

checkpoint_filename='desidlas/prediction/model/train_highsnr/current_99999'

dataset={}

for sight_id in r.keys():
    flux=np.array(r[sight_id]['FLUX'])
    (pred, conf, offset, coldensity)=predictions_ann(hyperparameters, flux, checkpoint_filename, TF_DEVICE='')
    dataset[sight_id]={'pred':pred,'conf':conf,'offset': offset, 'coldensity':coldensity }

np.save(savefile,dataset)
