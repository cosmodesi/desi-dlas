import numpy as np
import math
import re, os, traceback, sys, json
sys.path.append('/home/zjqi/data')
import argparse
import tensorflow as tf
import timeit
from tensorflow.python.framework import ops
from desidlas.datasets.get_flux import make_dataset
from desidlas.dla_cnn import defs
from tqdm import tqdm
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
    flux:list (400 length), flux from sightline
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






def predictions_desi(pred_sightlines,savefile):

    #parameters
    def _env_bool(name, default=False):
        value = os.environ.get(name)
        if value is None:
            return default
        return value == "1"

    def _low_input_config(bucket):
        bucket_env = bucket.upper()
        smooth = _env_bool(
            f"DESIDLAS_{bucket_env}_SMOOTH",
            _env_bool("DESIDLAS_LOW_SMOOTH", False),
        )
        input_size = int(os.environ.get(
            f"DESIDLAS_{bucket_env}_INPUT_SIZE",
            os.environ.get("DESIDLAS_LOW_INPUT_SIZE", defs.smooth_kernel if smooth else defs.kernel),
        ))
        matrix_size = int(os.environ.get(
            f"DESIDLAS_{bucket_env}_MATRIX_SIZE",
            os.environ.get("DESIDLAS_LOW_MATRIX_SIZE", 4 if smooth else 1),
        ))
        return input_size, matrix_size

    low1_input_size, low1_matrix_size = _low_input_config("low1")
    low2_input_size, low2_matrix_size = _low_input_config("low2")
    matrix_size={'high':1,'mid':1,'low1':low1_matrix_size,'low2':low2_matrix_size}
    INPUT_SIZE={'high':defs.kernel,'mid':defs.kernel,'low1':low1_input_size,'low2':low2_input_size}

    checkpoint_filename={
        'high': os.environ.get(
            'DESIDLAS_CKPT_HIGH',
            '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_highsnr/train_highsnr/current_99999',
        ),
        'mid': os.environ.get(
            'DESIDLAS_CKPT_MID',
            '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_midsnr/train_midsnr/current_99999',
        ),
        'low1': os.environ.get(
            'DESIDLAS_CKPT_LOW1',
            '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_lowsnr/train_lowsnr/current_99999',
        ),
        'low2': os.environ.get(
            'DESIDLAS_CKPT_LOW2',
            '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_lowsnr/train_lowsnr/current_99999',
        ),
    }
    

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)


    exception_counter = 0
    iteration_num = 0

    from dla_finder.training.parameterset import parameter_names
    from dla_finder.training.parameterset import parameters
    #from parameterset import parameter_names
    #from parameterset import parameters
    

    r=np.load(pred_sightlines,allow_pickle = True,encoding='latin1')
    


    dataset={}
    hyperparameters = {}


    TP=[]
    TN=[]
    FP=[]
    FN=[]
    #4 empty list to record number of TP,TN,FP,FN samples

    for sightline in tqdm(r.ravel()):
        if sightline.s2n<1.5:
            model='low1'
            for k in range(0,len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][0]
        elif sightline.s2n<3:
            model='low2'
            for k in range(0,len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][0]
        else:#s2n>3 use mid model
            model='mid'
            for k in range(0,len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][0]
        flux,lam=make_dataset(sightline, kernel=INPUT_SIZE[model], smooth=(matrix_size[model] > 1))
        '''
        elif sightline.s2n<6:
            #model='mid'
            #for k in range(0,len(parameter_names)):
                #hyperparameters[parameter_names[k]] = parameters[k][0]
        else:
            model='high'
            for k in range(0,len(parameter_names)):
                hyperparameters[parameter_names[k]] = parameters[k][1]
        '''
        (pred, conf, offset, coldensity)=predictions_ann(hyperparameters, INPUT_SIZE[model],matrix_size[model],flux, checkpoint_filename[model], TF_DEVICE='')#/gpu:1


        dataset[sightline.id]={'pred':pred,'conf':conf,'offset': offset, 'coldensity':coldensity, 'lam':lam }
        '''
        for p in range(0,len(pred)):
            if (r[sight_id]['labels_classifier'][p]==1) & (pred[p]==1):
                TP.append(p)
            if (r[sight_id]['labels_classifier'][p]==1) & (pred[p]==0):
                FN.append(p)
            if (r[sight_id]['labels_classifier'][p]==0) & (pred[p]==0):
                TN.append(p)
            if (r[sight_id]['labels_classifier'][p]==0) & (pred[p]==1):
                FP.append(p)
        '''

    np.save(savefile,dataset)
    print('done')
    '''
    print('samples of TP is %s'%(len(TP)))
    print('samples of TN is %s'%(len(TN)))
    print('samples of FP is %s'%(len(FP)))
    print('samples of FN is %s'%(len(FN)))
    '''
    

    
