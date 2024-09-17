import numpy as np
import math
import re, os, traceback, sys, json
sys.path.append('/global/cfs/cdirs/desi/users/jqzou')
import argparse
import tensorflow as tf
import logging
import timeit
from tensorflow.python.framework import ops
from desidlas.datasets.get_flux import make_dataset
from desidlas.parameters import kernel
from tqdm import tqdm
import multiprocessing
from desidlas.training.parameterset import parameter_names
from desidlas.training.parameterset import parameters
ops.reset_default_graph()



from desidlas.training.model import build_model
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

    #print("Localize Model processed {:d} samples in chunks of {:d} in {:0.1f} seconds".format(
    #      n_samples, BATCH_SIZE, timeit.default_timer() - timer))

    return pred, conf, offset, coldensity #return four labels

def pred_sightline(sightline):#sightline#pred_sightlines,savefile
    #sightline=np.load(pred_sightlines,allow_pickle = True,encoding='latin1').ravel()
    
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
        #np.save(savefile,dataset)
        return dataset

def execute_single_task(task_id, data_entries, savefile, cpu_count):
    with multiprocessing.Pool(cpu_count) as pool:
        results=pool.map(pred_sightline, data_entries)
        np.save(savefile,results)


def predictions_desi(pred_sightlines,savefile):

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    tf.get_logger().setLevel(logging.WARNING)
    exception_counter = 0
    iteration_num = 0
    
    '''
    r=np.load(pred_sightlines,allow_pickle = True,encoding='latin1')
    p=multiprocessing.Pool(processes=256)
    results=p.starmap(pred_sightline,tqdm([pred_sightlines,savefile]))
    np.save(savefile,results)
    
    '''
    total_cpu_count=256
    num_tasks=len(pred_sightlines)
    cpu_per_task = total_cpu_count // num_tasks
    processes = []
    for task_id in range(num_tasks):
        r=np.load(pred_sightlines[task_id],allow_pickle = True,encoding='latin1')
        p = multiprocessing.Process(target=execute_single_task, args=(task_id, tqdm(r.ravel()), savefile[task_id], cpu_per_task))
        processes.append(p)
        p.start()  

    for p in processes:
        p.join()
    
    
    #p=multiprocessing.Pool(processes=256)
    #if type(pred_sightlines)=='list':
    #for index in range(len(pred_sightlines)):
    #     results=p.map(pred_sightlines[index],tqdm(pred_sightlines.ravel()))
    #     np.save(savefile[index],results)    
    #else:
    #    r=np.load(pred_sightlines,allow_pickle = True,encoding='latin1')
    #    p=multiprocessing.Pool(processes=256)
    #    results=p.map(pred_sightline,tqdm(r.ravel()))
    #    np.save(savefile,results)
    
    
    
    
    
    '''
    r=np.load(pred_sightlines,allow_pickle = True,encoding='latin1')
    p=multiprocessing.Pool(processes=256)
    results=p.map(pred_sightline,tqdm(r.ravel()))
    
    #dataset[sightline.id]={'pred':pred,'conf':conf,'offset': offset, 'coldensity':coldensity, 'lam':lam }
            

    np.save(savefile,results)
    '''
    
    
   

    
