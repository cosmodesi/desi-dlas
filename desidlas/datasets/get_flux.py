import os
import scipy.signal as signal
import numpy as np
from desidlas.dla_cnn import defs
from desidlas.datasets.datasetting import split_sightline_into_samples

# Try to import CuPy
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def smooth_flux(flux):
    """Auto-select GPU or CPU implementation."""
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
    """GPU-accelerated version."""
    # Move to GPU
    flux_gpu = cp.asarray(flux)
    
    # Parallel median filters
    smooth3 = cp_signal.medfilt(flux_gpu, [1, 3])
    smooth7 = cp_signal.medfilt(flux_gpu, [1, 7])
    smooth15 = cp_signal.medfilt(flux_gpu, [1, 15])
    
    # Stack: (n_windows, 4, L)
    flux_matrix = cp.stack([flux_gpu, smooth3, smooth7, smooth15], axis=1)
    
    # Move back to CPU
    return cp.asnumpy(flux_matrix)


def smooth_flux_cpu(flux):
    """Vectorized CPU median smoothing over all windows.

    Input shape is [n_windows, n_pixels]. Output is [n_windows, 4, n_pixels].
    The previous implementation looped over windows in Python and called
    medfilt three times per window, which dominates low-SNR prediction time.
    """
    flux = np.asarray(flux, dtype=np.float32)
    smooth3 = signal.medfilt(flux, [1, 3])
    smooth7 = signal.medfilt(flux, [1, 7])
    smooth15 = signal.medfilt(flux, [1, 15])
    return np.stack([flux, smooth3, smooth7, smooth15], axis=1).astype(np.float32, copy=False)


def make_dataset(sightline, kernel=None, smooth=False):
    """Build prediction windows for one sightline.

    Parameters
    ----------
    kernel
        Window length. Defaults to the raw-flux kernel.
    smooth
        If True, return four channels: raw flux plus median filters 3, 7, 15.
    """
    if kernel is None:
        kernel = defs.kernel
    data_split = split_sightline_into_samples(
        sightline,
        REST_RANGE=defs.REST_RANGE,
        kernel=kernel,
        v=defs.best_v['all'],
    )
    flux = np.vstack([data_split[0]])
    if smooth:
        flux = smooth_flux(flux)
    
    input_lam = np.vstack([data_split[5]])
    
    return flux, input_lam
