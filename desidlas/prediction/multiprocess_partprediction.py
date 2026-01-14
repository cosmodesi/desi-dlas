# multiprocess_partprediction.py — GPU快速推理版(修正多维输入)
import os, re, sys, timeit, logging
import numpy as np
from tqdm import tqdm

# ---- 搜索路径 ----
sys.path.append('/global/cfs/cdirs/desi/users/jqzou')

# ---- TensorFlow TF1风格设置 ----
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.compat.v1 import ConfigProto

config = ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.intra_op_parallelism_threads = 2
config.inter_op_parallelism_threads = 2

# 线程数控制
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# ---- 项目依赖 ----
from desidlas.datasets.get_flux import make_dataset
from desidlas.training.parameterset import parameter_names, parameters
from desidlas.training.model import build_model

# ---- 模型常量(全局) ----
MATRIX_SIZE = {'high': 1, 'mid': 1, 'low': 4}
INPUT_SIZE  = {'high': 400, 'mid': 400, 'low': 600}
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

# ---- 工具函数 ----
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

# ---- 全局会话缓存 ----
_SESSION_CACHE = {}

def _get_model_session(model_key, INPUT_SIZE, matrix_size, ckpt_path):
    """每种模型只加载一次"""
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

# ---- 流式批处理推理(支持多维输入) ----
def _infer_bucket_stream(lines, index_list, model_key, INPUT_SIZE, matrix_size, 
                         ckpt_path, batch_size=16384):
    """
    对同一分桶的多条sightline进行流式批处理推理
    支持 2D: [batch, L] 和 3D: [batch, C, L] 输入
    """
    t0 = timeit.default_timer()
    g, sess, (x, keep_prob, t_pred, t_conf, t_off, t_col) = _get_model_session(
        model_key, INPUT_SIZE, matrix_size, ckpt_path
    )

    L = INPUT_SIZE
    C = matrix_size  # 通道数: low=4, mid/high=1
    
    # 根据matrix_size决定缓冲区形状
    if C > 1:
        buf = np.empty((batch_size, C, L), dtype=np.float32)  # 3D: [batch, 4, 600]
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
            flux_i, lam_i = make_dataset(lines[i])
            if flux_i is None or flux_i.size == 0:
                results[i] = None
                continue
                
            fi = flux_i.astype('float32', copy=False)
            Wi = fi.shape[0]
            tmp_lam[i] = lam_i

            # 验证形状匹配
            if C > 1:
                assert fi.ndim == 3 and fi.shape[1:] == (C, L), \
                    f"Expected shape [W, {C}, {L}], got {fi.shape}"
            else:
                assert fi.ndim == 2 and fi.shape[1] == L, \
                    f"Expected shape [W, {L}], got {fi.shape}"

            start = 0
            while start < Wi:
                need = min(batch_size - pos, Wi - start)
                
                # 复制数据到缓冲区(处理多维情况)
                if C > 1:
                    buf[pos:pos+need, :, :] = fi[start:start+need, :, :]
                else:
                    buf[pos:pos+need, :] = fi[start:start+need, :]
                
                pending.append((i, pos, pos+need))
                pos += need
                start += need

                # 批满则运行
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

        # 处理最后一批
        if pos > 0 and pending:
            # 只取实际填充的部分
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

        # 合并结果
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

# ---- 单文件快速推理 ----
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

    # 流式批处理
    out_map = {}
    if idx_low1:
        out_map.update(_infer_bucket_stream(
            lines, idx_low1, 'low1',
            INPUT_SIZE['low'], MATRIX_SIZE['low'], CKPT['low1'],
            batch_size=16384
        ))
    if idx_low2:
        out_map.update(_infer_bucket_stream(
            lines, idx_low2, 'low2',
            INPUT_SIZE['low'], MATRIX_SIZE['low'], CKPT['low2'],
            batch_size=16384
        ))
    if idx_mid:
        out_map.update(_infer_bucket_stream(
            lines, idx_mid, 'mid',
            INPUT_SIZE['mid'], MATRIX_SIZE['mid'], CKPT['mid'],
            batch_size=16384
        ))

    # 还原顺序
    results = [out_map.get(i, None) for i in range(len(lines))]
    
    # 确保目录存在
    os.makedirs(os.path.dirname(savefile), exist_ok=True)
    np.save(savefile, results)
    
    print(f"[FAST] Saved to {savefile} in {timeit.default_timer()-t0:.2f}s", flush=True)

# ---- 顶层调度 ----
def predictions_desi(pred_sightlines, savefile):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel(logging.ERROR)

    if isinstance(pred_sightlines, str):
        pred_file_fast(pred_sightlines, savefile)
        return

    assert len(pred_sightlines) == len(savefile), "路径列表长度不一致"
    for in_path, out_path in zip(pred_sightlines, savefile):
        pred_file_fast(in_path, out_path)
