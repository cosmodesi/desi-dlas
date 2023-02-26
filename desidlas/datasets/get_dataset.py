import numpy as np
import scipy.signal as signal
from desidlas.datasets.datasetting import split_sightline_into_samples,select_samples_50p_pos_neg,pad_sightline
from desidlas.datasets.preprocess import label_sightline
from desidlas.dla_cnn.spectra_utils import get_lam_data
from desidlas.dla_cnn import defs
REST_RANGE = defs.REST_RANGE
kernel = defs.kernel
smooth_kernel= defs.smooth_kernel
best_v = defs.best_v

def smooth_flux(flux):
    """
    Smooth flux using median filter.

    Parameters:
    -----------------------------------------------
    flux: np.ndarry flux for every kernel

    Return:
    -----------------------------------------------
    flux_matrix:list, 2-dimension flux data after smoothing

    """
    flux_matrix=[]
    for sample in flux:
        smooth3=signal.medfilt(sample,3)
        smooth7=signal.medfilt(sample,7)
        smooth15=signal.medfilt(sample,15)
        flux_matrix.append(np.array([sample,smooth3,smooth7,smooth15]))
    return flux_matrix

def make_dataset(sightline,REST_RANGE=REST_RANGE, v=best_v['all']):
    if sightline.s2n>3:
        data_split=split_sightline_into_samples(sightline,kernel=kernel,REST_RANGE=REST_RANGE, v=v)
        if data_split[0] is None : # empty
            return None,None
        flux=np.vstack([data_split[0]])
    else:
        data_split=split_sightline_into_samples(sightline,kernel=smooth_kernel,REST_RANGE=REST_RANGE, v=v)
        if data_split[0] is None : # empty
            return None,None
        flux=np.vstack([data_split[0]])
        flux=np.array(smooth_flux(flux))
    input_lam=np.vstack([data_split[5]])

    return flux,input_lam

def make_training(sightlines, kernel=kernel, REST_RANGE=REST_RANGE, v=best_v['all'],output=None):
    """
    Generate training sets for DESI.

    Parameters:
    -----------------------------------------------
    sightlines: list of 'dla_cnn.data_model.Sightline' object, the sightlines should be preprocessed.

    Returns
    -----------------------------------------------
    dataset:dict, the training set contains flux and 3 labels, the validation set contains flux, lam, 3 labels and DLAs' data.

    """
    dataset={}
    for sightline in sightlines:
        wavelength_dlas=[dla.central_wavelength for dla in sightline.dlas]
        coldensity_dlas=[dla.col_density for dla in sightline.dlas]
        label_sightline(sightline, kernel=kernel, REST_RANGE=REST_RANGE)
        data_split=split_sightline_into_samples(sightline,REST_RANGE=REST_RANGE, kernel=kernel,v=v)
        sample_masks=select_samples_50p_pos_neg(sightline, kernel=kernel)
        if sample_masks !=[]:
            flux=np.vstack([data_split[0][m] for m in sample_masks])
            labels_classifier=np.hstack([data_split[1][m] for m in sample_masks])
            labels_offset=np.hstack([data_split[2][m] for m in sample_masks])
            col_density=np.hstack([data_split[3][m] for m in sample_masks])
            dataset[sightline.id]={'FLUX':flux,'labels_classifier':labels_classifier,'labels_offset':labels_offset,'col_density': col_density}
    np.save(output,dataset)
    return dataset


def make_smoothtraining(sightlines,kernel=smooth_kernel, REST_RANGE=REST_RANGE, v=best_v['all'], output=None):
    """
    Generate smoothed training set or validation set for DESI.

    Parameters:
    -----------------------------------------------
    sightlines: list of 'dla_cnn.data_model.Sightline' object, the sightlines should be preprocessed.


    Returns
    -----------------------------------------------
    dataset:dict, the training set contains smoothed flux and 3 labels, the validation set contains smoothed flux, lam, 3 labels and DLAs' data.

    """
    dataset={}
    for sightline in sightlines:
        wavelength_dlas=[dla.central_wavelength for dla in sightline.dlas]
        coldensity_dlas=[dla.col_density for dla in sightline.dlas]
        label_sightline(sightline, kernel=kernel, REST_RANGE=REST_RANGE)
        data_split=split_sightline_into_samples(sightline, REST_RANGE=REST_RANGE, kernel=kernel,v=v)
        sample_masks=select_samples_50p_pos_neg(sightline,kernel=kernel)
        if sample_masks !=[]:
            flux=np.vstack([data_split[0][m] for m in sample_masks])
            labels_classifier=np.hstack([data_split[1][m] for m in sample_masks])
            labels_offset=np.hstack([data_split[2][m] for m in sample_masks])
            col_density=np.hstack([data_split[3][m] for m in sample_masks])
            flux_matrix=smooth_flux(flux)
            dataset[sightline.id]={'FLUX':flux_matrix,'labels_classifier':labels_classifier,'labels_offset':labels_offset,'col_density': col_density}
    np.save(output,dataset)
    return dataset
