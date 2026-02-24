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

# ---- Global cache: load each model once ----
_model_cache = {}

def _get_model_sess(model_key, INPUT_SIZE, matrix_size, checkpoint_filename):
    """
    Returns (graph, sess, handles):
      handles = (x, keep_prob, t_pred, t_conf, t_off, t_col)
    """
    if model_key in _model_cache:
        return _model_cache[model_key]

    g = tf.Graph()
    with g.as_default():
        # Build graph
        build_model(hyperparameters=None, INPUT_SIZE=INPUT_SIZE, matrix_size=matrix_size)
        # Note: use TF1 Session with config to avoid grabbing all GPU memory at once
        sess = tf.compat.v1.Session(graph=g, config=config)
        with sess.as_default():
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, checkpoint_filename + ".ckpt")
            x         = g.get_tensor_by_name('x:0')
            keep_prob = g.get_tensor_by_name('keep_prob:0')
            t_pred    = g.get_tensor_by_name('prediction:0')
            t_conf    = g.get_tensor_by_name('output_classifier:0')
            t_off     = g.get_tensor_by_name('y_nn_offset:0')
            t_col     = g.get_tensor_by_name('y_nn_coldensity:0')

    _model_cache[model_key] = (g, sess, (x, keep_prob, t_pred, t_conf, t_off, t_col))
    return _model_cache[model_key]


def _infer_batch(model_key, INPUT_SIZE, matrix_size, ckpt, flux_batch, batch_size=8192):
    """
    Use cached session to run batch inference on flux_batch (shape [N, L]).
    Returns pred, conf, offset, coldensity (all shape [N]).
    """
    g, sess, (x, keep_prob, t_pred, t_conf, t_off, t_col) = _get_model_sess(
        model_key, INPUT_SIZE, matrix_size, ckpt
    )
    n = flux_batch.shape[0]
    out_pred = np.empty(n, np.float32)
    out_conf = np.empty(n, np.float32)
    out_off  = np.empty(n, np.float32)
    out_col  = np.empty(n, np.float32)
    with g.as_default():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            p, c, o, d = sess.run(
                [t_pred, t_conf, t_off, t_col],
                feed_dict={x: flux_batch[sl, :], keep_prob: 1.0}
            )
            out_pred[sl], out_conf[sl], out_off[sl], out_col[sl] = p, c, o, d
    return out_pred, out_conf, out_off, out_col




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




'''
def predictions_ann(hyperparameters, INPUT_SIZE,matrix_size,flux, checkpoint_filename, TF_DEVICE=''):
    
    #Perform training
    #Parameters
    #----------
    #hyperparameters:hyperparameters for the CNN model structure
    #flux:list (400 length), flux from sightline
    #checkpoint_filename: CNN model file used to detect DLAs
    #TF_DEVICE: use which gpu to train, default is '/gpu:1'

    #Returns
    #-------
    #pred:0 or 1, label for every window, 0 means no DLA in this window and 1 means this window has a DLA
    #conf:[0,1], confidence level, label for every window, pred is 0 when conf is below the critical value (0.5 default), pred is 1 when conf is above the critical value
    #offset: [-60,+60] , label for every window, pixel numbers between DLA center and the window center
    #coldensity:label for every window, the estimated NHI column density

    

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
'''
def predictions_ann(hyperparameters, INPUT_SIZE, matrix_size, flux, checkpoint_filename, TF_DEVICE=''):
    """
    Run inference with a trained CNN (TF1-style).
    """
    timer = timeit.default_timer()
    BATCH_SIZE = 4000
    n_samples = flux.shape[0]

    pred = np.zeros((n_samples,), dtype=np.float32)
    conf = np.copy(pred)
    offset = np.copy(pred)
    coldensity = np.copy(pred)

    # Key: build graph -> build Session (with ConfigProto(allow_growth=True))
    with tf.Graph().as_default():
        build_model(hyperparameters, INPUT_SIZE, matrix_size)

        # Note: Session must use config=config, or it may grab all GPU memory
        with tf.device(TF_DEVICE), tf.compat.v1.Session(config=config) as sess:
            # Restore checkpoint
            tf.compat.v1.train.Saver().restore(sess, checkpoint_filename + ".ckpt")

            # Batch inference
            for i in range(0, n_samples, BATCH_SIZE):
                sl = slice(i, min(i + BATCH_SIZE, n_samples))
                pred[sl], conf[sl], offset[sl], coldensity[sl] = sess.run(
                    [t('prediction'), t('output_classifier'), t('y_nn_offset'), t('y_nn_coldensity')],
                    feed_dict={t('x'): flux[sl, :],
                               t('keep_prob'): 1.0}
                )

    return pred, conf, offset, coldensity

def pred_sightline(sightline):#sightline#pred_sightlines,savefile
    #sightline=np.load(pred_sightlines,allow_pickle = True,encoding='latin1').ravel()
    
    #parameters
    matrix_size={'high':1,'mid':1,'low1':1,'low2':1}
    INPUT_SIZE={'high':400,'mid':400,'low1':400,'low2':400}

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
    hyperparameters = {}
    if sightline != []:
        flux,lam=make_dataset(sightline)
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
        (pred, conf, offset, coldensity)=predictions_ann(hyperparameters, INPUT_SIZE[model],matrix_size[model],flux,checkpoint_filename[model], TF_DEVICE='')#/gpu:1
        dataset={'pred':pred,'conf':conf,'offset': offset, 'coldensity':coldensity, 'lam':lam }
        return dataset


def pred_file_fast(npy_path, savefile):
    """
    Fast path for a single sightline file (.npy):
      - group by SNR bucket (low1/low2/mid)
      - concatenate windows per group and run batched inference
      - split back per sightline and keep original results structure
    """
    # Parameters consistent with the original logic
    matrix_size = {'high':1, 'mid':1, 'low1':1, 'low2':1}
    INPUT_SIZE  = {'high':400,'mid':400,'low1':400,'low2':400}
    ckpt = {
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

    arr = np.load(npy_path, allow_pickle=True, encoding='latin1')
    lines = arr.ravel()

    # Group by s2n
    idx_low1, idx_low2, idx_mid = [], [], []
    for i, sight in enumerate(lines):
        if sight == []:  # skip empty
            continue
        s2n = getattr(sight, 's2n', 0)
        if s2n < 1.5:
            idx_low1.append(i)
        elif s2n < 3:
            idx_low2.append(i)
        else:
            idx_mid.append(i)

    results = [None] * len(lines)

    # Helper: pack a group of sightlines into a large batch
    def build_batch(index_list, model_key):
        flux_list, lam_list, sizes = [], [], []
        for i in index_list:
            flux, lam = make_dataset(lines[i])   # original function
            flux_list.append(flux.astype('float32'))  # shape [Wi, L]
            lam_list.append(lam)
            sizes.append(flux.shape[0])
        if len(flux_list) == 0:
            return None, None, None
        big = np.vstack(flux_list)   # [sum Wi, L]
        return big, lam_list, sizes

    # Low1 SNR group
    big, lam_list, sizes = build_batch(idx_low1, 'low1')
    if big is not None:
        p, c, o, d = _infer_batch('low1', INPUT_SIZE['low1'], matrix_size['low1'], ckpt['low1'], big)
        # Split back per sightline
        cursor = 0
        for i, sz in zip(idx_low1, sizes):
            sl = slice(cursor, cursor+sz)
            results[i] = {
                'pred': p[sl], 'conf': c[sl],
                'offset': o[sl], 'coldensity': d[sl],
                'lam': lam_list[cursor - (cursor - cursor)]  # matching element from lam_list
            }
            cursor += sz

    # Low2 SNR group
    big, lam_list, sizes = build_batch(idx_low2, 'low2')
    if big is not None:
        p, c, o, d = _infer_batch('low2', INPUT_SIZE['low2'], matrix_size['low2'], ckpt['low2'], big)
        cursor = 0
        for i, sz in zip(idx_low2, sizes):
            sl = slice(cursor, cursor+sz)
            results[i] = {
                'pred': p[sl], 'conf': c[sl],
                'offset': o[sl], 'coldensity': d[sl],
                'lam': lam_list[cursor - (cursor - cursor)]
            }
            cursor += sz

    # Mid SNR group
    big, lam_list, sizes = build_batch(idx_mid, 'mid')
    if big is not None:
        p, c, o, d = _infer_batch('mid', INPUT_SIZE['mid'], matrix_size['mid'], ckpt['mid'], big)
        cursor = 0
        for i, sz in zip(idx_mid, sizes):
            sl = slice(cursor, cursor+sz)
            results[i] = {
                'pred': p[sl], 'conf': c[sl],
                'offset': o[sl], 'coldensity': d[sl],
                'lam': lam_list[cursor - (cursor - cursor)]
            }
            cursor += sz

    np.save(savefile, results)


'''
def execute_single_task(task_id, data_entries, savefile, cpu_count):
    with multiprocessing.Pool(cpu_count) as pool:
        results=pool.map(pred_sightline, data_entries)
        pool.close()
        pool.join() 
        np.save(savefile,results)
'''
def execute_single_task(task_id, data_entries, savefile, cpu_count):
    # Single-process serial: avoid TF + Pool deadlocks/stalls
    results = [pred_sightline(x) for x in data_entries]
    np.save(savefile, results)

'''
def predictions_desi(pred_sightlines,savefile):

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    tf.get_logger().setLevel(logging.WARNING)
    exception_counter = 0
    iteration_num = 0



    total_cpu_count=256
    num_tasks=len(pred_sightlines)
    cpu_per_task = total_cpu_count // num_tasks

    
    processes = []
    
    if type(pred_sightlines)==str:
        r=np.load(pred_sightlines,allow_pickle = True,encoding='latin1')
        results=pred_sightline(tqdm(r.ravel()))
        np.save(savefile,results)
    else:
        for task_id in range(num_tasks):
            r=np.load(pred_sightlines[task_id],allow_pickle = True,encoding='latin1')
            p = multiprocessing.Process(target=execute_single_task, args=(task_id, tqdm(r.ravel()), savefile[task_id], cpu_per_task))
            processes.append(p)
            p.start()  
        for p in processes:
            p.join()
'''    
'''
def predictions_desi(pred_sightlines, savefile):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    tf.get_logger().setLevel(logging.WARNING)

    # Single-process sequential run (GPU is most stable; throughput depends on batch size)
    if isinstance(pred_sightlines, str):
        r = np.load(pred_sightlines, allow_pickle=True, encoding='latin1')
        results = pred_sightline(tqdm(r.ravel()))
        np.save(savefile, results)
    else:
        assert len(pred_sightlines) == len(savefile), "pred_sightlines and savefile lengths differ"
        for task_id, path in enumerate(pred_sightlines):
            r = np.load(path, allow_pickle=True, encoding='latin1')
            results = [pred_sightline(x) for x in tqdm(r.ravel(), total=r.size)]
            np.save(savefile[task_id], results)
'''    
    
def predictions_desi(pred_sightlines, savefile):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(logging.ERROR)

    # Single file path -> run directly
    if isinstance(pred_sightlines, str):
        r = np.load(pred_sightlines, allow_pickle=True, encoding='latin1')
        results = [pred_sightline(x) for x in tqdm(r.ravel(), total=r.size)]
        np.save(savefile, results)
        return

    # Multiple file paths -> run sequentially
    assert len(pred_sightlines) == len(savefile), "pred_sightlines and savefile lengths differ"
    for in_path, out_path in zip(pred_sightlines, savefile):
        r = np.load(in_path, allow_pickle=True, encoding='latin1')
        results = [pred_sightline(x) for x in tqdm(r.ravel(), total=r.size)]
        np.save(out_path, results)


    
