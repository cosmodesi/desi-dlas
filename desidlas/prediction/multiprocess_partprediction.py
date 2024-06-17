import numpy as np
import math
import re, os, traceback, sys, json
sys.path.append('/global/cfs/cdirs/desi/users/jqzou')
import argparse
import tensorflow as tf
import timeit
from tensorflow.python.framework import ops
from dla_finder.datasets.get_flux import make_dataset
from dla_finder.parameters import kernel
from tqdm import tqdm
import multiprocessing
from dla_finder.training.parameterset import parameter_names
from dla_finder.training.parameterset import parameters
ops.reset_default_graph()



from dla_finder.training.model import build_model
#from model import build_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
tensor_regex = re.compile('.*:\d*')
# Get a tensor by name, convenience method
def t(tensor_name):
    tensor_name = tensor_name+":0" if not tensor_regex.match(tensor_name) else tensor_name
    return tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)





def predictions_ann(hyperparameters, INPUT_SIZE,matrix_size,flux, checkpoint_filename, TF_DEVICE=''):
    '''
    Perform training
    Parameters
    ----------
    hyperparameters:hyperparameters for the CNN model structure
    flux:list (400 or 600 length), flux from sightline
    checkpoint_filename: CNN model file used to detect DLAs
    TF_DEVICE: use which gpu to train, default is '/gpu:1'

    Returns
    -------
    pred:0 or 1, label for every window, 0 means no DLA in this window and 1 means this window has a DLA
    conf:[0,1], confidence level, label for every window, pred is 0 when conf is below the critical value (0.5 default), pred is 1 when conf is above the critical value
    offset: [-60,+60] , label for every window, pixel numbers between DLA center and the window center
    coldensity:label for every window, the estimated NHI column density

    '''

    timer = timeit.default_timer() # import timer to record time used for every prediction
    BATCH_SIZE = 4000
    n_samples = flux.shape[0]
    pred = np.zeros((n_samples,), dtype=np.float32)
    conf = np.copy(pred)
    offset = np.copy(pred)
    coldensity = np.copy(pred) #establish 4 empty list to save label values


    with tf.Graph().as_default():
        build_model(hyperparameters,INPUT_SIZE,matrix_size) # build the CNN model according to hyperparameters

        with tf.device(TF_DEVICE), tf.compat.v1.Session() as sess:
            tf.compat.v1.train.Saver().restore(sess, checkpoint_filename+".ckpt") #load model files
            for i in range(0,n_samples,BATCH_SIZE):
                pred[i:i+BATCH_SIZE], conf[i:i+BATCH_SIZE], offset[i:i+BATCH_SIZE], coldensity[i:i+BATCH_SIZE] = \
                    sess.run([t('prediction'), t('output_classifier'), t('y_nn_offset'), t('y_nn_coldensity')],
                             feed_dict={t('x'):                 flux[i:i+BATCH_SIZE,:],
                                        t('keep_prob'):         1.0}) #get prediction labels

    print("Localize Model processed {:d} samples in chunks of {:d} in {:0.1f} seconds".format(
          n_samples, BATCH_SIZE, timeit.default_timer() - timer))

    return pred, conf, offset, coldensity #return four labels

def pred_sightline(sightline):
    #parameters
    matrix_size={'high':1,'mid':1,'low':4}
    INPUT_SIZE={'high':400,'mid':400,'low':600}

    checkpoint_filename={'high':'/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_highsnr/train_highsnr/current_99999','mid':'/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_midsnr/train_midsnr/current_99999','low':'/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_lowsnr/train_lowsnr/current_99999'}
    hyperparameters = {}
    if sightline != []:
        flux,lam=make_dataset(sightline)
        if sightline.s2n<3:
            model='low'
            for k in range(0,len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][0]
        else:#s2n>3 use mid model
            model='mid'
            for k in range(0,len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][0]
        (pred, conf, offset, coldensity)=predictions_ann(hyperparameters, INPUT_SIZE[model],matrix_size[model],flux,checkpoint_filename[model], TF_DEVICE='')#/gpu:1
        dataset={'pred':pred,'conf':conf,'offset': offset, 'coldensity':coldensity, 'lam':lam }
        return dataset



def predictions_desi(pred_sightlines,savefile):

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)


    exception_counter = 0
    iteration_num = 0

    r=np.load(pred_sightlines,allow_pickle = True,encoding='latin1')
    p=multiprocessing.Pool(processes=30)
    results=p.map(pred_sightline,tqdm(r.ravel()))
    
    #dataset[sightline.id]={'pred':pred,'conf':conf,'offset': offset, 'coldensity':coldensity, 'lam':lam }
            

    np.save(savefile,results)
   

    
