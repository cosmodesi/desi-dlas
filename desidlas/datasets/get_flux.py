import scipy.signal as signal
import numpy as np
from desidlas.parameters import kernel
from .input_set import split_sightline_into_samples

def smooth_flux(flux):
    flux_matrix=[]
    for sample in flux:
        smooth3=signal.medfilt(sample,3)
        smooth7=signal.medfilt(sample,7)
        smooth15=signal.medfilt(sample,15)
        flux_matrix.append(np.array([sample,smooth3,smooth7,smooth15]))
    return flux_matrix
    
def make_dataset(sightline):
    if sightline.s2n>3:
        data_split=split_sightline_into_samples(sightline,kernel=kernel['highsnr'])
        flux=np.vstack([data_split[0]])
    else:
        data_split=split_sightline_into_samples(sightline,kernel=kernel['lowsnr'])
        flux=np.vstack([data_split[0]])
        flux=np.array(smooth_flux(flux))
     #labels_classifier=np.hstack([data_split[1]])
     #labels_offset=np.hstack([data_split[2]])
     #col_density=np.hstack([data_split[3]])
    input_lam=np.vstack([data_split[5]])
    
    
    return flux,input_lam
    
    