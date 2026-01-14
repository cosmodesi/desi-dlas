import os
import scipy.signal as signal
import numpy as np
from desidlas.parameters import kernel
from .input_set import split_sightline_into_samples

# 尝试导入CuPy
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def smooth_flux(flux):
    """自动选择GPU或CPU版本"""
    if _gpu_available() and flux.shape[0] > 100:  # 只在数据量大时用GPU
        return smooth_flux_gpu(flux)
    else:
        return smooth_flux_cpu(flux)


def _gpu_available():
    if not HAS_GPU:
        return False
    if os.environ.get("DESIDLAS_FORCE_CPU") == "1":
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def smooth_flux_gpu(flux):
    """GPU加速版本"""
    # 转到GPU
    flux_gpu = cp.asarray(flux)
    
    # 并行做中值滤波
    smooth3 = cp_signal.medfilt(flux_gpu, [1, 3])
    smooth7 = cp_signal.medfilt(flux_gpu, [1, 7])
    smooth15 = cp_signal.medfilt(flux_gpu, [1, 15])
    
    # 堆叠: (n_windows, 4, 400)
    flux_matrix = cp.stack([flux_gpu, smooth3, smooth7, smooth15], axis=1)
    
    # 转回CPU
    return cp.asnumpy(flux_matrix)


def smooth_flux_cpu(flux):
    """原始CPU版本（保持不变）"""
    flux_matrix = []
    for sample in flux:
        smooth3 = signal.medfilt(sample, 3)
        smooth7 = signal.medfilt(sample, 7)
        smooth15 = signal.medfilt(sample, 15)
        flux_matrix.append(np.array([sample, smooth3, smooth7, smooth15]))
    return np.array(flux_matrix)


def make_dataset(sightline):
    """保持接口不变"""
    if sightline.s2n > 3:
        data_split = split_sightline_into_samples(sightline, kernel=kernel['highsnr'])
        flux = np.vstack([data_split[0]])
    else:
        data_split = split_sightline_into_samples(sightline, kernel=kernel['lowsnr'])
        flux = np.vstack([data_split[0]])
        flux = smooth_flux(flux)  # 自动选择GPU或CPU
    
    input_lam = np.vstack([data_split[5]])
    
    return flux, input_lam
