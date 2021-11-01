#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:40:41 2020

@author: benwang
"""

""" Methods for CNN DLA Finder Prediction"""
#!python

"""
0. Update all methods to TF 2.1 including using tf.Dataset
"""
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
    INPUT_SIZE: pixels numbers for each window , 400 for high SNR and 600 for low SNR
    matrix_size: 1 if without smoothing, 4 if smoothing for low SNR
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






if __name__ == '__main__':


    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--preddataset', help='Datasets to detect DLAs , npy format', required=True, default=False)
    parser.add_argument('-o', '--output_file', help='output files to save the prediction result, npy format', required=False, default=False)
    parser.add_argument('-model', '--modelfiles', help='CNN models for prediction, high snr model or mid snr model', required=False, default=False)
    parser.add_argument('-t', '--INPUT_SIZE', help='set the input data size', required=False, default=400)
    parser.add_argument('-m', '--matrix_size', help='set the matrix size when using smooth', required=False, default=1)

    args = vars(parser.parse_args())

    
    batch_results_file = args['output_file']
    INPUT_SIZE = args['INPUT_SIZE']
    matrix_size = args['matrix_size']

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)


    exception_counter = 0
    iteration_num = 0

    from desidlas.training.parameterset import parameter_names
    from desidlas.training.parameterset import parameters
    hyperparameters = {}
    


    pred_dataset=args['preddataset']
    savefile=args['output_file']

    r=np.load(pred_dataset,allow_pickle = True,encoding='latin1').item()

    modelfile=args['modelfiles']
    if modelfile == 'high':
        checkpoint_filename='desidlas/prediction/model/train_highsnr/current_99999'
        for k in range(0,len(parameter_names)):
            hyperparameters[parameter_names[k]] = parameters[k][1]
    if modelfile == 'mid':
        checkpoint_filename='desidlas/prediction/model/train_midsnr/current_99999'
        for k in range(0,len(parameter_names)):
            hyperparameters[parameter_names[k]] = parameters[k][1]
    

    dataset={}


    TP=[]
    TN=[]
    FP=[]
    FN=[]
    #4 empty list to record number of TP,TN,FP,FN samples

    for sight_id in r.keys():
    
        flux=np.array(r[sight_id]['FLUX'])

        (pred, conf, offset, coldensity)=predictions_ann(hyperparameters, INPUT_SIZE,matrix_size,flux, checkpoint_filename, TF_DEVICE='')


        dataset[sight_id]={'pred':pred,'conf':conf,'offset': offset, 'coldensity':coldensity }

        for p in range(0,len(pred)):
            if (r[sight_id]['labels_classifier'][p]==1) & (pred[p]==1):
                TP.append(p)
            if (r[sight_id]['labels_classifier'][p]==1) & (pred[p]==0):
                FN.append(p)
            if (r[sight_id]['labels_classifier'][p]==0) & (pred[p]==0):
                TN.append(p)
            if (r[sight_id]['labels_classifier'][p]==0) & (pred[p]==1):
                FP.append(p)
        

    np.save(savefile,dataset)

    print('samples of TP is %s'%(len(TP)))
    print('samples of TN is %s'%(len(TN)))
    print('samples of FP is %s'%(len(FP)))
    print('samples of FN is %s'%(len(FN)))
    

    
