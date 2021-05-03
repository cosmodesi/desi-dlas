""" Code to build/load/write DESI Training sets"""

'''
1. Load up the Sightlines
2. Split into samples of kernel length
3. Grab DLAs and non-DLA samples
4. Hold in memory or write to disk??
5. Convert to TF Dataset
'''


import itertools

import numpy as np

from desidlas.dla_cnn.spectra_utils import get_lam_data
from desidlas.dla_cnn.defs import REST_RANGE,kernel,best_v
    
def pad_sightline(sightline, lam, lam_rest, ix_dla_range,kernelrangepx,v=best_v['b']):
        """
    padding the left and right sides of the spectra

    Parameters
    ----------
    sightline: dla_cnn.data_model.Sightline
    lam: np.ndarray
    lam_rest: np.ndarray
    ix_dla_range: np.ndarray   Indices listing where to search for the DLA
    kernelrangepx:int, half of the kernel
    v:float, best v for the b band

    Returns
    flux_padded:np.ndarray,flux after padding
    lam_padded:np.ndarray,lam after padding
    pixel_num_left:int,the number of pixels padded to the left side of spectra 
    -------

    """
    c = 2.9979246e8
    dlnlambda = np.log(1+v/c)
    #pad left side
    if np.nonzero(ix_dla_range)[0][0]<kernelrangepx:
        pixel_num_left=kernelrangepx-np.nonzero(ix_dla_range)[0][0]
        pad_lam_left= lam[0]*np.exp(dlnlambda*np.array(range(-pixel_num_left,0)))
        pad_value_left = np.mean(sightline.flux[0:50])
    else:
        pixel_num_left=0
        pad_lam_left=[]
        pad_value_left=[] 
    #pad right side
    if np.nonzero(ix_dla_range)[0][-1]>len(lam)-kernelrangepx:
        pixel_num_right=kernelrangepx-(len(lam)-np.nonzero(ix_dla_range)[0][-1])
        pad_lam_right= lam[0]*np.exp(dlnlambda*np.array(range(len(lam),len(lam)+pixel_num_right)))
        pad_value_right = np.mean(sightline.flux[-50:])
    else:
        pixel_num_right=0
        pad_lam_right=[]
        pad_value_right=[]
    flux_padded = np.hstack((pad_lam_left*0+pad_value_left, sightline.flux,pad_lam_right*0+pad_value_right))
    lam_padded = np.hstack((pad_lam_left,lam,pad_lam_right))
    return flux_padded,lam_padded,pixel_num_left

def split_sightline_into_samples(sightline, REST_RANGE=REST_RANGE, kernel=kernel):
    """
    Split the sightline into a series of snippets, each with length kernel

    Parameters
    ----------
    sightline: dla_cnn.data_model.Sightline
    REST_RANGE: list
    kernel: int, optional

    Returns
    -------

    """
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    kernelrangepx = int(kernel/2) #200
    #samplerangepx = int(kernel*pos_sample_kernel_percent/2) #60
    #padding the sightline:
    flux_padded,lam_padded,pixel_num_left=pad_sightline(sightline,lam,lam_rest,ix_dla_range,kernelrangepx,v=best_v['b'])
     
    #ix_dlas = [(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) for dla in sightline.dlas]
    #coldensity_dlas = [dla.col_density for dla in sightline.dlas]       # column densities matching ix_dlas

    # FLUXES - Produce a 1748x400 matrix of flux values
    #fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0])))
    fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(flux_padded), np.nonzero(ix_dla_range)[0]+pixel_num_left)))
    lam_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(lam_padded), np.nonzero(ix_dla_range)[0]+pixel_num_left)))
    #using cut will lose side information,so we use padding instead of cutting 
    #fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0][cut])))
    #lam_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(lam), np.nonzero(ix_dla_range)[0][cut])))
    #the wavelength and flux array we input:
    input_lam=lam_padded[np.nonzero(ix_dla_range)[0]+pixel_num_left]
    input_flux=flux_padded[np.nonzero(ix_dla_range)[0]+pixel_num_left]
    # Return
    return fluxes_matrix, sightline.classification, sightline.offsets, sightline.column_density,lam_matrix,input_lam,input_flux
    #return fluxes_matrix, sightline.classification, sightline.offsets, sightline.column_density

def select_samples_50p_pos_neg(sightline,kernel=kernel):
    """
    For a given sightline, generate the indices for DLAs and for without
    Split 50/50 to have equal representation

    Parameters
    ----------
    classification: np.ndarray
        Array of classification values.  1=DLA; 0=Not; -1=not analyzed

    Returns
    -------
    idx: np.ndarray
        positive + negative indices

    """
    #classification = data[1]
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso)
    kernelrangepx = int(kernel/2)
    #cut=((np.nonzero(ix_dla_range)[0])>=kernelrangepx)&((np.nonzero(ix_dla_range)[0])<=(len(lam)-kernelrangepx-1))
    #newclassification=sightline.classification[cut]
    num_pos = np.sum(sightline.classification==1, dtype=np.float64)
    num_neg = np.sum(sightline.classification==0, dtype=np.float64)
    n_samples = int(min(num_pos, num_neg))

    r = np.random.permutation(len(sightline.classification))

    pos_ixs = r[sightline.classification[r]==1][0:n_samples]
    neg_ixs = r[sightline.classification[r]==0][0:n_samples]
    # num_total = data[0].shape[0]
    # ratio_neg = num_pos / num_neg

    # pos_mask = classification == 1      # Take all positive samples

    # neg_ixs_by_ratio = np.linspace(1,num_total-1,round(ratio_neg*num_total), dtype=np.int32) # get all samples by ratio
    # neg_mask = np.zeros((num_total),dtype=np.bool) # create a 0 vector of negative samples
    # neg_mask[neg_ixs_by_ratio] = True # set the vector to positives, selecting for the appropriate ratio across the whole sightline
    # neg_mask[pos_mask] = False # remove previously positive samples from the set
    # neg_mask[classification == -1] = False # remove border samples from the set, what remains is still in the right ratio

    # return pos_mask | neg_mask
    return np.hstack((pos_ixs,neg_ixs))
    #return np.hstack(pos_ixs,neg_ixs)

