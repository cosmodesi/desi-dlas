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

def make_datasets(sightlines, kernel=kernel, REST_RANGE=REST_RANGE, v=best_v['all'], output=None, validate=True, chunk_size=100):
    """
    Generate training set or validation set for DESI.

    Parameters:
    -----------------------------------------------
    sightlines: list of 'dla_cnn.data_model.Sightline' object, the sightlines should be preprocessed.
    validate: bool, optional, this decides whether to add wavelength in the dataset
    validate: bool, optional, this decides whether to smooth the flux. In our paper, we smooth the flux for spectra with SNR<3
    chunk_size: int, optional, how many sightlines to save once

    Returns
    -----------------------------------------------
    dataset: dict, the training set contains flux and 3 labels, the validation set contains flux, lam, 3 labels and DLAs' data.
    """
    dataset = {}
    count = 0  # 计数处理过的 sightline
    file_idx = 0  # 文件编号

    for sightline in sightlines:
        wavelength_dlas = [dla.central_wavelength for dla in sightline.dlas]
        coldensity_dlas = [dla.col_density for dla in sightline.dlas]   
        label_sightline(sightline, kernel=kernel, REST_RANGE=REST_RANGE)
        data_split = split_sightline_into_samples(sightline, REST_RANGE=REST_RANGE, kernel=kernel, v=v)
        
        if validate:
            flux = np.vstack([data_split[0]])
            labels_classifier = np.hstack([data_split[1]])
            labels_offset = np.hstack([data_split[2]])
            col_density = np.hstack([data_split[3]])
            lam = np.vstack([data_split[4]])
            dataset[sightline.id] = {
                'FLUX': flux,
                'lam': lam,
                'labels_classifier': labels_classifier,
                'labels_offset': labels_offset,
                'col_density': col_density,
                'wavelength_dlas': wavelength_dlas,
                'coldensity_dlas': coldensity_dlas
            }
        else:
            sample_masks = select_samples_50p_pos_neg(sightline, kernel=kernel)
            # 判断 sample_masks 非空, 这里根据实际类型调整判断方式：
            if len(sample_masks) > 0:
                flux = np.vstack([data_split[0][m] for m in sample_masks])
                labels_classifier = np.hstack([data_split[1][m] for m in sample_masks])
                labels_offset = np.hstack([data_split[2][m] for m in sample_masks])
                col_density = np.hstack([data_split[3][m] for m in sample_masks])
                dataset[sightline.id] = {
                    'FLUX': flux,
                    'labels_classifier': labels_classifier,
                    'labels_offset': labels_offset,
                    'col_density': col_density
                }

        count += 1
        # 达到chunk_size，保存并清空dataset
        if count >= chunk_size:
            outpath = f"{output}_{file_idx}.npy"
            np.save(outpath, dataset)
            print(f"Saved {count} sightlines to {outpath}")
            dataset = {}
            count = 0
            file_idx += 1

    # 保存剩余未保存的数据
    if dataset:
        outpath = f"{output}_{file_idx}.npy"
        np.save(outpath, dataset)
        print(f"Saved final {count} sightlines to {outpath}")

    return dataset



        
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

#smooth flux for low S/N sightlines
def make_smoothdatasets(sightlines,kernel=smooth_kernel, REST_RANGE=REST_RANGE, v=best_v['all'], output=None, validate=True):
    """
    Generate smoothed training set or validation set for DESI.
    
    Parameters:
    -----------------------------------------------
    sightlines: list of 'dla_cnn.data_model.Sightline' object, the sightlines should be preprocessed.
    validate: bool
    
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
        if validate:
            flux=np.vstack([data_split[0]])
            labels_classifier=np.hstack([data_split[1]])
            labels_offset=np.hstack([data_split[2]])
            col_density=np.hstack([data_split[3]])
            lam=np.vstack([data_split[4]])
            flux_matrix=smooth_flux(flux)
            dataset[sightline.id]={'FLUX':flux_matrix,'lam':lam,'labels_classifier':  labels_classifier, 'labels_offset':labels_offset , 'col_density': col_density,'wavelength_dlas':wavelength_dlas,'coldensity_dlas':coldensity_dlas} 
        else:
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