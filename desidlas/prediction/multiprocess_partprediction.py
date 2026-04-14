# multiprocess_partprediction.py — GPU fast inference (multi-dim input fix)
import os, re, sys, timeit, logging
import numpy as np
from tqdm import tqdm

# ---- Search path ----
sys.path.append('/global/cfs/cdirs/desi/users/jqzou')

# ---- TensorFlow TF1-style settings ----
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.intra_op_parallelism_threads = 2
config.inter_op_parallelism_threads = 2

# Thread count controls
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ---- Project dependencies ----
from desidlas.datasets.get_flux import make_dataset
from desidlas.training.parameterset import parameter_names, parameters
from desidlas.training.model import build_model
from desidlas.dla_cnn import defs

# ---- Model constants (global) ----
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
    return {"input_size": input_size, "matrix_size": matrix_size, "smooth": matrix_size > 1}


MODEL_INPUT = {
    'low1': _low_input_config('low1'),
    'low2': _low_input_config('low2'),
    'mid': {'input_size': defs.kernel, 'matrix_size': 1, 'smooth': False},
}
CKPT = {
    'low1': os.environ.get(
        'DESIDLAS_CKPT_LOW1',
        '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_lowsnr/train_lowsnr/current_99999',
    ),
    'low2': os.environ.get(
        'DESIDLAS_CKPT_LOW2',
        '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_lowsnr/train_lowsnr/current_99999',
    ),
    'mid': os.environ.get(
        'DESIDLAS_CKPT_MID',
        '/global/cfs/cdirs/desi/users/jqzou/dla_finder/prediction/model/train_midsnr/train_midsnr/current_99999',
    ),
}

# ---- Helpers ----
def _get_handles(graph):
    x = graph.get_tensor_by_name('x:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    t_pred = graph.get_tensor_by_name('prediction:0')
    t_conf = graph.get_tensor_by_name('output_classifier:0')
    t_off = graph.get_tensor_by_name('y_nn_offset:0')
    t_col = graph.get_tensor_by_name('y_nn_coldensity:0')
    return x, keep_prob, t_pred, t_conf, t_off, t_col

def get_hparams(model: str):
    hp = {}
    for k in range(len(parameter_names)):
        hp[parameter_names[k]] = parameters[k][0]
    return hp

# ---- Global session cache ----
_SESSION_CACHE = {}

def _get_model_session(model_key, INPUT_SIZE, matrix_size, ckpt_path):
    """Load each model once."""
    if model_key in _SESSION_CACHE:
        print(f"[MODEL] reuse cached → {model_key}", flush=True)
        return _SESSION_CACHE[model_key]

    print(f"[MODEL] build+restore start → {model_key} ({ckpt_path})", flush=True)
    hparams = get_hparams(model_key)

    g = tf.Graph()
    with g.as_default():
        build_model(hyperparameters=hparams, INPUT_SIZE=INPUT_SIZE, matrix_size=matrix_size)
        print(f"[MODEL] graph built → {model_key}", flush=True)
        sess = tf.compat.v1.Session(graph=g, config=config)
        with sess.as_default():
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, ckpt_path + ".ckpt")
        print(f"[MODEL] checkpoint restored → {model_key}", flush=True)
        handles = _get_handles(g)

    _SESSION_CACHE[model_key] = (g, sess, handles)
    return _SESSION_CACHE[model_key]

# ---- Streaming batch inference (supports multi-dim input) ----
def _infer_bucket_stream(lines, index_list, model_key, INPUT_SIZE, matrix_size, 
                         ckpt_path, batch_size=16384):
    """
    Streamed batch inference for sightlines in the same bucket.
    Supports 2D: [batch, L] and 3D: [batch, C, L] inputs.
    """
    t0 = timeit.default_timer()
    g, sess, (x, keep_prob, t_pred, t_conf, t_off, t_col) = _get_model_session(
        model_key, INPUT_SIZE, matrix_size, ckpt_path
    )

    L = INPUT_SIZE
    C = matrix_size  # channels: low/mid/high=1 (raw flux)
    
    # Decide buffer shape based on matrix_size
    if C > 1:
        buf = np.empty((batch_size, C, L), dtype=np.float32)
    else:
        buf = np.empty((batch_size, L), dtype=np.float32)     # 2D: [batch, 400]
    
    results = {i: None for i in index_list}
    tmp_pred = {i: [] for i in index_list}
    tmp_conf = {i: [] for i in index_list}
    tmp_off  = {i: [] for i in index_list}
    tmp_col  = {i: [] for i in index_list}
    tmp_lam  = {i: None for i in index_list}

    print(f"[STREAM] Start {model_key.upper()} batch inference for {len(index_list)} sightlines (shape: {buf.shape})", flush=True)
    
    with g.as_default():
        pos = 0
        pending = []
        total_runs = 0
        
        for i in tqdm(index_list, desc=f"{model_key.upper()}", leave=False):
            flux_i, lam_i = make_dataset(
                lines[i],
                kernel=INPUT_SIZE,
                smooth=(matrix_size > 1),
            )
            if flux_i is None or flux_i.size == 0:
                results[i] = None
                continue
                
            fi = flux_i.astype('float32', copy=False)
            Wi = fi.shape[0]
            tmp_lam[i] = lam_i

            # Validate shape
            if C > 1:
                assert fi.ndim == 3 and fi.shape[1:] == (C, L), \
                    f"Expected shape [W, {C}, {L}], got {fi.shape}"
            else:
                assert fi.ndim == 2 and fi.shape[1] == L, \
                    f"Expected shape [W, {L}], got {fi.shape}"

            start = 0
            while start < Wi:
                need = min(batch_size - pos, Wi - start)
                
                # Copy data into buffer (handle multi-dim)
                if C > 1:
                    buf[pos:pos+need, :, :] = fi[start:start+need, :, :]
                else:
                    buf[pos:pos+need, :] = fi[start:start+need, :]
                
                pending.append((i, pos, pos+need))
                pos += need
                start += need

                # Run when batch is full
                if pos == batch_size:
                    p, c, o, d = sess.run(
                        [t_pred, t_conf, t_off, t_col],
                        feed_dict={x: buf, keep_prob: 1.0}
                    )
                    for (idx, s, e) in pending:
                        tmp_pred[idx].append(p[s:e])
                        tmp_conf[idx].append(c[s:e])
                        tmp_off[idx].append(o[s:e])
                        tmp_col[idx].append(d[s:e])
                    pos = 0
                    pending.clear()
                    total_runs += 1

        # Handle final partial batch
        if pos > 0 and pending:
            # Only use the filled portion
            actual_buf = buf[:pos, ...] if C > 1 else buf[:pos, :]
            
            p, c, o, d = sess.run(
                [t_pred, t_conf, t_off, t_col],
                feed_dict={x: actual_buf, keep_prob: 1.0}
            )
            for (idx, s, e) in pending:
                tmp_pred[idx].append(p[s:e])
                tmp_conf[idx].append(c[s:e])
                tmp_off[idx].append(o[s:e])
                tmp_col[idx].append(d[s:e])
            total_runs += 1

        # Merge results
        for i in index_list:
            if not tmp_pred[i]:
                results[i] = None
                continue
            pred = np.concatenate(tmp_pred[i], axis=0)
            conf = np.concatenate(tmp_conf[i], axis=0)
            off  = np.concatenate(tmp_off[i],  axis=0)
            col  = np.concatenate(tmp_col[i],  axis=0)
            results[i] = {
                'pred': pred, 'conf': conf,
                'offset': off, 'coldensity': col,
                'lam': tmp_lam[i]
            }

    print(f"[STREAM] {model_key.upper()} done in {timeit.default_timer()-t0:.2f}s ({total_runs} GPU runs)", flush=True)
    return results

# ---- Single-file fast inference ----
def pred_file_fast(npy_path, savefile):
    t0 = timeit.default_timer()
    print(f"[FAST] Processing: {npy_path}", flush=True)
    
    arr = np.load(npy_path, allow_pickle=True, encoding='latin1')
    lines = arr.ravel()
    print(f"[FAST] Loaded {len(lines)} sightlines", flush=True)

    idx_low1, idx_low2, idx_mid = [], [], []
    for i, sight in enumerate(lines):
        if sight == []:
            continue
        s2n = getattr(sight, 's2n', 0)
        if s2n < 1.5:
            idx_low1.append(i)
        elif s2n < 3:
            idx_low2.append(i)
        else:
            idx_mid.append(i)
    
    print(f"[FAST] Groups: low1={len(idx_low1)}, low2={len(idx_low2)}, mid={len(idx_mid)}", flush=True)
    print(
        f"[FAST] Input config: "
        f"low1={MODEL_INPUT['low1']['input_size']}x{MODEL_INPUT['low1']['matrix_size']}, "
        f"low2={MODEL_INPUT['low2']['input_size']}x{MODEL_INPUT['low2']['matrix_size']}, "
        f"mid={MODEL_INPUT['mid']['input_size']}x{MODEL_INPUT['mid']['matrix_size']}",
        flush=True,
    )

    # Streaming batch processing
    out_map = {}
    if idx_low1:
        out_map.update(_infer_bucket_stream(
            lines, idx_low1, 'low1',
            MODEL_INPUT['low1']['input_size'], MODEL_INPUT['low1']['matrix_size'], CKPT['low1'],
            batch_size=16384
        ))
    if idx_low2:
        out_map.update(_infer_bucket_stream(
            lines, idx_low2, 'low2',
            MODEL_INPUT['low2']['input_size'], MODEL_INPUT['low2']['matrix_size'], CKPT['low2'],
            batch_size=16384
        ))
    if idx_mid:
        out_map.update(_infer_bucket_stream(
            lines, idx_mid, 'mid',
            MODEL_INPUT['mid']['input_size'], MODEL_INPUT['mid']['matrix_size'], CKPT['mid'],
            batch_size=16384
        ))

    # Restore original order
    results = [out_map.get(i, None) for i in range(len(lines))]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    np.save(savefile, results)
    
    print(f"[FAST] Saved to {savefile} in {timeit.default_timer()-t0:.2f}s", flush=True)

# ---- Top-level dispatch ----
def predictions_desi(pred_sightlines, savefile):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(logging.ERROR)

    if isinstance(pred_sightlines, str):
        pred_file_fast(pred_sightlines, savefile)
        return

    assert len(pred_sightlines) == len(savefile), "Path list lengths do not match"
    workers = int(os.environ.get("DESIDLAS_CPU_WORKERS", "1"))
    if workers <= 1:
        for in_path, out_path in zip(pred_sightlines, savefile):
            pred_file_fast(in_path, out_path)
        return

    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
    except Exception:
        import multiprocessing as mp
        ctx = mp

    args = list(zip(pred_sightlines, savefile))
    with ctx.Pool(processes=workers) as pool:
        list(pool.starmap(pred_file_fast, args))
