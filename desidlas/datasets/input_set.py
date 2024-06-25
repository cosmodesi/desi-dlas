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
from desidlas.parameters import REST_RANGE,kernel,best_v,camera,continuum_model,smooth_model


def get_lam_data(loglam, z_qso, REST_RANGE):
    """
    Generate wavelengths from the log10 wavelengths

    Parameters
    ----------
    loglam: np.ndarray
    z_qso: float
    REST_RANGE: list
        Lowest rest wavelength to search, highest rest wavelength,  number of pixels in the search

    Returns
    -------
    lam: np.ndarray
    lam_rest: np.ndarray
    ix_dla_range: np.ndarray
        Indices listing where to search for the DLA
    """
    lam = 10.0 ** loglam
    lam_rest = lam / (1.0 + z_qso)
    ix_dla_range = np.logical_and(lam_rest >= REST_RANGE[0], lam_rest <= REST_RANGE[1])  # &(lam>=3800)#np.logical_and(lam_index>=kernelrangepx,lam_index<=len(lam)-kernelrangepx-1)

    return lam, lam_rest, ix_dla_range


def pad_sightline(sightline, lam, ix_dla_range,kernelrangepx,v,continuum):
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
    if continuum:
        cont_padded = np.hstack(
            (pad_lam_left * 0 + pad_value_left, sightline.continuum, pad_lam_right * 0 + pad_value_right))
        normalize_flux = flux_padded / cont_padded
        normalize_flux[np.isnan(normalize_flux)] = 1
        return normalize_flux, lam_padded, pixel_num_left
    else:
        return flux_padded,lam_padded,pixel_num_left


    
def split_sightline_into_samples(sightline, REST_RANGE=REST_RANGE, kernel=kernel[smooth_model],v=best_v[camera],continuum=continuum_model):
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
    kernelrangepx = int(kernel/2)

    flux_padded, lam_padded, pixel_num_left = pad_sightline(sightline, lam, ix_dla_range, kernelrangepx,
                                                                v,continuum)

     
    #ix_dlas = [(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) for dla in sightline.dlas]
    #coldensity_dlas = [dla.col_density for dla in sightline.dlas]       # column densities matching ix_dlas

    # FLUXES - Produce a 1748x400 matrix of flux values
    #fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0]))) nersc we can not use np.vstack
    fluxes_matrix = np.array(list(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(flux_padded), np.nonzero(ix_dla_range)[0]+pixel_num_left))))
    lam_matrix = np.array(list(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(lam_padded), np.nonzero(ix_dla_range)[0]+pixel_num_left))))
    #using cut will lose side information,so we use padding instead of cutting 
    #fluxes_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0][cut])))
    #lam_matrix = np.vstack(map(lambda x:x[0][x[1]-kernelrangepx:x[1]+kernelrangepx],zip(itertools.repeat(lam), np.nonzero(ix_dla_range)[0][cut])))
    #the wavelength and flux array we input:
    input_lam=lam_padded[np.nonzero(ix_dla_range)[0]+pixel_num_left]
    input_flux=flux_padded[np.nonzero(ix_dla_range)[0]+pixel_num_left]
    # Return
    return fluxes_matrix, sightline.classification, sightline.offsets, sightline.column_density,lam_matrix,input_lam,input_flux
    #return fluxes_matrix, sightline.classification, sightline.offsets, sightline.column_density

