""" Methods for training the Model"""
#!python

"""
0. Update all methods to TF 2.1 including using tf.Dataset
"""
import numpy as np
import math
import re, os, traceback, sys, json
import argparse
import tensorflow as tf
import os
from pathlib import Path

from tensorflow.python.framework import ops
ops.reset_default_graph()

from desidlas.training.model import build_model

#Parameter configuration for tensorflow session
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True #Dynamically apply for video memory, slowly increase gpu capacity from less to more
config.allow_soft_placement=True  #Allow tensorflow to automatically allocate devices
config.log_device_placement = True #Record the log of which device each node is assigned to, for the convenience of debugging


tensor_regex = re.compile('.*:\d*')

# Get a tensor by name, convenience method
def t(tensor_name):
    '''
    Parameters
    ----------
    tensor_name: str, the name for each tensor
    
    Returns
    ----------
    tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name):establish a tensor, make a default graph according to the tensor name
    '''
    tensor_name = tensor_name+":0" if not tensor_regex.match(tensor_name) else tensor_name
    return tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)


# Called from train_ann to perform a test of the train or test data, needs to separate pos/neg to get accurate #'s
def train_ann_test_batch(sess, ixs, data, summary_writer=None):  
    #inputs: label_classifier, label_offset, label_coldensity
    """
    Perform training and estimate accuracy,RMSE after every iteration training
    
    Parameters
    ----------
    sess:tf.Session object
    ixs:np.ndarray, indices for each window in the dataset
    data:sightline data in training and testing dataset
    summary_writer: whether to save the training log file at given path, default is true
    
    Returns
    -------
    classifier_accuracy: the classification accuracy after this training iteration
    classifier_loss_value: the classification RMSE after this training iteration
    result_rmse_offset:the offset RMSE after this training iteration
    result_loss_offset_regression:the offset loss function value after this training iteration
    result_rmse_coldensity:the column density RMSE after this training iteration
    result_loss_coldensity_regression:the column density loss function after this training iteration

    """
    MAX_BATCH_SIZE = 40000.0
    classifier_accuracy = 0.0
    classifier_loss_value = 0.0
    result_rmse_offset = 0.0
    result_loss_offset_regression = 0.0
    result_rmse_coldensity = 0.0
    result_loss_coldensity_regression = 0.0

    # split x and labels into pos/neg
    pos_ixs = ixs[data['labels_classifier'][ixs]==1]
    neg_ixs = ixs[data['labels_classifier'][ixs]==0]
    pos_ixs_split = np.array_split(pos_ixs, math.ceil(len(pos_ixs)/MAX_BATCH_SIZE)) if len(pos_ixs) > 0 else []
    neg_ixs_split = np.array_split(neg_ixs, math.ceil(len(neg_ixs)/MAX_BATCH_SIZE)) if len(neg_ixs) > 0 else []

    sum_samples = float(len(ixs))       # forces cast of percent len(pos_ix)/sum_samples to a float value
    sum_pos_only = float(len(pos_ixs))

    # Process pos and neg samples separately
    # pos
    for pos_ix in pos_ixs_split:
        tensors = [t('accuracy'), t('loss_classifier'), t('rmse_offset'), t('loss_offset_regression'),
                        t('rmse_coldensity'), t('loss_coldensity_regression'), t('global_step')]
        if summary_writer is not None:
            tensors.extend(tf.compat.v1.get_collection('SUMMARY_A')) # adding summary list to every tensor, used to save the training log
            tensors.extend(tf.compat.v1.get_collection('SUMMARY_B'))
            tensors.extend(tf.compat.v1.get_collection('SUMMARY_C'))
            tensors.extend(tf.compat.v1.get_collection('SUMMARY_shared'))
        #start training, runing the graph in tensorflow according to the tensors and data
        run = sess.run(tensors, feed_dict={t('x'):                data['fluxes'][pos_ix],
                                           t('label_classifier'): data['labels_classifier'][pos_ix],
                                           t('label_offset'):     data['labels_offset'][pos_ix],
                                           t('label_coldensity'): data['col_density'][pos_ix],
                                           t('keep_prob'):        1.0})
        weight_wrt_total_samples = len(pos_ix)/sum_samples
        weight_wrt_pos_samples = len(pos_ix)/sum_pos_only
        # print "DEBUG> Processing %d positive samples" % len(pos_ix), weight_wrt_total_samples, sum_samples, weight_wrt_pos_samples, sum_pos_only
        classifier_accuracy += run[0] * weight_wrt_total_samples
        classifier_loss_value += np.sum(run[1]) * weight_wrt_total_samples
        result_rmse_offset += run[2] * weight_wrt_total_samples
        result_loss_offset_regression += run[3] * weight_wrt_total_samples
        result_rmse_coldensity += run[4] * weight_wrt_pos_samples
        result_loss_coldensity_regression += run[5] * weight_wrt_pos_samples

       

    # neg
    for neg_ix in neg_ixs_split:
        tensors = [t('accuracy'), t('loss_classifier'), t('rmse_offset'), t('loss_offset_regression'), t('global_step')]
        if summary_writer is not None:
            tensors.extend(tf.compat.v1.get_collection('SUMMARY_A'))
            tensors.extend(tf.compat.v1.get_collection('SUMMARY_B'))
            tensors.extend(tf.compat.v1.get_collection('SUMMARY_shared'))

        run = sess.run(tensors, feed_dict={t('x'):                data['fluxes'][neg_ix],
                                           t('label_classifier'): data['labels_classifier'][neg_ix],
                                           t('label_offset'):     data['labels_offset'][neg_ix],
                                           t('keep_prob'):        1.0})
        weight_wrt_total_samples = len(neg_ix)/sum_samples
        # print "DEBUG> Processing %d negative samples" % len(neg_ix), weight_wrt_total_samples, sum_samples
        classifier_accuracy += run[0] * weight_wrt_total_samples
        classifier_loss_value += np.sum(run[1]) * weight_wrt_total_samples
        result_rmse_offset += run[2] * weight_wrt_total_samples
        result_loss_offset_regression += run[3] * weight_wrt_total_samples


        

    return classifier_accuracy, classifier_loss_value, \
           result_rmse_offset, result_loss_offset_regression, \
           result_rmse_coldensity, result_loss_coldensity_regression



def train_ann(hyperparameters, train_dataset, test_dataset, save_filename=None, load_filename=None, tblogs = "../tmp/tblogs", TF_DEVICE='/gpu:1'):
    """
    Perform training

    Parameters
    ----------
    hyperparameters:hyperparameters for the CNN model structure
    train_dataset: the dataset used for training
    test_dataset: the dataset used for testing
    save_filename: the path to save the model file, ckpt format
    load_filename: load a model file,none for initial training
    tblogs : a path to save tblog files necessary for tf.summary
    TF_DEVICE: use which gpu to train, default is '/gpu:1'

    Returns
    -------
    best_accuracy:the best classification accuracy during the training
    test_accuracy: the final test accuracy
    np.mean(loss_value):mean value of the loss function
    best_offset_rmse: the minimum value of offset RMSE
    result_rmse_offset:the final value of offset RMSE
    best_density_rmse:the minimum value of column density RMSE
    result_rmse_coldensity:the final value of column density RMSE

    """
    training_iters = hyperparameters['training_iters']
    batch_size = hyperparameters['batch_size']
    dropout_keep_prob = hyperparameters['dropout_keep_prob']

    # Predefine variables that need to be returned from local scope
    best_accuracy = 0.0
    best_offset_rmse = 999999999
    best_density_rmse = 999999999
    test_accuracy = None
    loss_value = None
    record_file=open('1016snr5smnewtrain.txt','w')

    with tf.Graph().as_default():
        train_step_ABC, tb_summaries = build_model(hyperparameters,INPUT_SIZE,matrix_size)

        with tf.device('/gpu:1'), tf.compat.v1.Session(config=config) as sess:
            # Restore or initialize model
            if load_filename is not None:
                tf.train.Saver().restore(sess, load_filename+".ckpt")
                
            else:
               
                sess.run(tf.compat.v1.global_variables_initializer())

            
            summary_writer = tf.summary.create_file_writer(tblogs)

            for i in range(training_iters):
                # Grab a batch
                batch_fluxes, batch_labels_classifier, batch_labels_offset, batch_col_density = train_dataset.next_batch(batch_size)
                # Train
                sess.run(train_step_ABC, feed_dict={t('x'):                batch_fluxes,
                                                    t('label_classifier'): batch_labels_classifier,
                                                    t('label_offset'):     batch_labels_offset,
                                                    t('label_coldensity'): batch_col_density,
                                                    t('keep_prob'):        dropout_keep_prob})

                if i % 200 == 0:   # i%2 = 1 on 200ths iteration, that's important so we have the full batch pos/neg
                    train_accuracy, loss_value, result_rmse_offset, result_loss_offset_regression, \
                    result_rmse_coldensity, result_loss_coldensity_regression \
                        = train_ann_test_batch(sess, np.random.permutation(train_dataset.fluxes.shape[0])[0:10000],
                                               train_dataset.data, summary_writer=summary_writer)
                       
                    
                    print("step {:06d}, classify-offset-density acc/loss - RMSE/loss      {:0.3f}/{:0.3f} - {:0.3f}/{:0.3f} - {:0.3f}/{:0.3f}".format(
                          i, train_accuracy, float(np.mean(loss_value)), result_rmse_offset,
                             result_loss_offset_regression, result_rmse_coldensity,
                             result_loss_coldensity_regression))
                    record_file.write("step {:06d}, classify-offset-density acc/loss - RMSE/loss      {:0.3f}/{:0.3f} - {:0.3f}/{:0.3f} - {:0.3f}/{:0.3f}\n".format(
                          i, train_accuracy, float(np.mean(loss_value)), result_rmse_offset,
                             result_loss_offset_regression, result_rmse_coldensity,
                             result_loss_coldensity_regression))
                    
                # Run on test
                if i % 5000 == 0 or i == training_iters - 1:
                    test_accuracy, _, result_rmse_offset, _, result_rmse_coldensity, _ = train_ann_test_batch(
                        sess, np.arange(test_dataset.fluxes.shape[0]), test_dataset.data)
                    best_accuracy = test_accuracy if test_accuracy > best_accuracy else best_accuracy
                    best_offset_rmse = result_rmse_offset if result_rmse_offset < best_offset_rmse else best_offset_rmse
                    best_density_rmse = result_rmse_coldensity if result_rmse_coldensity < best_density_rmse else best_density_rmse
                    
                    print("             test accuracy/offset RMSE/density RMSE:     {:0.3f} / {:0.3f} / {:0.3f}\n".format(
                          test_accuracy, result_rmse_offset, result_rmse_coldensity))
                    record_file.write("             test accuracy/offset RMSE/density RMSE:     {:0.3f} / {:0.3f} / {:0.3f}".format(
                          test_accuracy, result_rmse_offset, result_rmse_coldensity))
                    
                    save_checkpoint(sess, (save_filename + "_" + str(i)) if save_filename is not None else None)

           # Save checkpoint
            save_checkpoint(sess, save_filename)

            return best_accuracy, test_accuracy, np.mean(loss_value), best_offset_rmse, result_rmse_offset, \
                   best_density_rmse, result_rmse_coldensity

def save_checkpoint(sess, save_filename):
    """
    save model file

    Parameters
    ----------
    sess:tf.Session object
    save_filename:the path to save the model

    Returns
    -------

    """
    if save_filename is not None:
        tf.compat.v1.train.Saver().save(sess, save_filename + ".ckpt")
        with open(checkpoint_filename + "_hyperparams.json", 'w') as fp:
            json.dump(hyperparameters, fp)
        print("Model saved in file: %s" % save_filename + ".ckpt")

def calc_normalized_score(best_accuracy, best_offset_rmse, best_coldensity_rmse):
    """
    Generate the normalized score from the 3 loss functions

    This should be used for a hyperparameter search

    Parameters
    ----------
    best_accuracy:the highest classification accuracy during the training
    best_offset_rmse:the minimum offset RMSE during the training
    best_coldensity_rmse:the minimum column density during the training

    Returns:
    -------
    normalized_score:used to select hyperparameters. A better hyperparameter combine will have a smaller normalized_score

    """
    # These mean & std dev are used to normalize the score from all 3 loss functions for hyperparam optimize
    # TODO -- Reconsider these for DESI
    mean_best_accuracy = 0.02
    std_best_accuracy = 0.0535
    mean_best_offset_rmse = 4.0
    std_best_offset_rmse = 1.56
    mean_best_coldensity_rmse = 0.33
    std_best_coldensity_rmse = 0.1
    normalized_score = (1 - best_accuracy - mean_best_accuracy) / std_best_accuracy + \
                       (best_offset_rmse - mean_best_offset_rmse) / std_best_offset_rmse + \
                       (best_coldensity_rmse - mean_best_coldensity_rmse) / std_best_coldensity_rmse
    
    return normalized_score



if __name__ == '__main__':
    #
    # Execute batch mode
    #
    from desidlas.data_model.Dataset import Dataset
    
    datafile_path = os.path.join(resource_filename('desidlas', 'tests'), 'datafile')
    traindata_path=os.path.join(datafile_path, 'sightlines-16-1375.npy')
    testdata_path=os.path.join(datafile_path, 'sightlines-16-1375.npy')
    
    save_path=os.path.join(resource_filename('desidlas', 'training'), 'model_files')
    savemodel_path=os.path.join(save_path, 'train_highsnr','current')
    savebatch_path=os.path.join(save_path, 'train_highsnr','batch_results.csv')
    
    DEFAULT_TRAINING_ITERS = 100000 #30000
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--hyperparamsearch', help='Run hyperparam search', required=False, action='store_true', default=False)
    parser.add_argument('-o', '--output_file', help='Hyperparam csv result file location', required=False, default=savebatch_path)
    parser.add_argument('-i', '--iterations', help='Number of training iterations', required=False, default=DEFAULT_TRAINING_ITERS)
    parser.add_argument('-l', '--loadmodel', help='Specify a model name to load', required=False, default=None)
    parser.add_argument('-c', '--checkpoint_file', help='Name of the checkpoint file to save (without file extension)', required=False, default=savemodel_path) #../models/training/current
    parser.add_argument('-r', '--train_dataset_filename', help='File name of the training dataset without extension', required=False, default=traindata_path)
    parser.add_argument('-e', '--test_dataset_filename', help='File name of the testing dataset without extension', required=False, default=testdata_path)
    parser.add_argument('-t', '--INPUT_SIZE', help='set the input data size', required=False, default=600)
    parser.add_argument('-m', '--matrix_size', help='set the matrix size when using smooth', required=False, default=4)
    args = vars(parser.parse_args())

    RUN_SINGLE_ITERATION = not args['hyperparamsearch']
    checkpoint_filename = args['checkpoint_file'] if RUN_SINGLE_ITERATION else None
    batch_results_file = args['output_file']
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    # set the input data size
    INPUT_SIZE = args['INPUT_SIZE']
    matrix_size = args['matrix_size']

    train_dataset = Dataset(args['train_dataset_filename'])
    test_dataset = Dataset(args['test_dataset_filename'])

    exception_counter = 0
    iteration_num = 0


    
    # Write out CSV header
    os.remove(batch_results_file) if os.path.exists(batch_results_file) else None
    with open(batch_results_file, "a") as csvoutput:
        csvoutput.write("iteration_num,normalized_score,best_accuracy,last_accuracy,last_objective,best_offset_rmse,last_offset_rmse,best_coldensity_rmse,last_coldensity_rmse," + ",".join(parameter_names) + "\n")


    from desidlas.training.parameterset import parameter_names
    from desidlas.training.parameterset import parameters
    
    #hyperparameter search
   
    hyperparameters = {}
    #choose the hyperparameters
    for k in range(0,len(parameter_names)):
        hyperparameters[parameter_names[k]] = parameters[k][0]

    #start the training
    (best_accuracy, last_accuracy, last_objective, best_offset_rmse, last_offset_rmse, best_coldensity_rmse,
    last_coldensity_rmse) = train_ann(hyperparameters, train_dataset, test_dataset,
                                    save_filename=checkpoint_filename, load_filename=args['loadmodel'])

    
