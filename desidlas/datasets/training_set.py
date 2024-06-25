""" Module to vette results against Human catalogs
  SDSS-DR5 (JXP) and BOSS (Notredaeme)
"""

import numpy as np
from desidlas.parameters import REST_RANGE,kernel,smooth_model,pos_sample_kernel_percent

def label_sightline(sightline, kernel=kernel[smooth_model], REST_RANGE=REST_RANGE, pos_sample_kernel_percent=pos_sample_kernel_percent):#0.3
    """
    Add labels to input sightline based on the DLAs along that sightline

    Parameters
    ----------
    sightline: dla_cnn.data_model.Sightline
    pos_sample_kernel_percent: float
    kernel: int
    REST_RANGE: list

    Returns
    -------
    classification: np.ndarray
        is 1 / 0 / -1 for DLA/nonDLA/border
    offsets_array: np.ndarray
        offset
    column_density: np.ndarray

    """
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    samplerangepx = int(kernel*pos_sample_kernel_percent/2)
    ix_dlas=[]
    coldensity_dlas=[]
    for dla in sightline.dlas:
        if (REST_RANGE[0]<(dla.central_wavelength/(1+sightline.z_qso))<REST_RANGE[1]):
            ix_dlas.append(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin())
            coldensity_dlas.append(dla.col_density)    # column densities matching ix_dlas

    '''
    # FLUXES - Produce a 1748x400 matrix of flux values
    fluxes_matrix = np.vstack(map(lambda f,r:f[r-kernelrangepx:r+kernelrangepx],
                                  zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0])))
    '''

    # CLASSIFICATION (1 = positive sample, 0 = negative sample, -1 = border sample not used
    # Start with all samples zero
    classification = np.zeros((np.sum(ix_dla_range)), dtype=np.float32)
    # overlay samples that are too close to a known DLA, write these for all DLAs before overlaying positive sample 1's
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx*2:ix_dla+samplerangepx*2+1] = -1
        # Mark out Ly-B areas
        lyb_ix = sightline.get_lyb_index(ix_dla)
        classification[lyb_ix-samplerangepx:lyb_ix+samplerangepx+1] = -1
    # mark out bad samples from custom defined markers
    #for marker in sightline.data_markers:
        #assert marker.marker_type == Marker.IGNORE_FEATURE              # we assume there are no other marker types for now
        #ixloc = np.abs(lam_rest - marker.lam_rest_location).argmin()
        #classification[ixloc-samplerangepx:ixloc+samplerangepx+1] = -1
    # overlay samples that are positive
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx:ix_dla+samplerangepx+1] = 1

    # OFFSETS & COLUMN DENSITY
    offsets_array = np.full([np.sum(ix_dla_range)], np.nan, dtype=np.float32)     # Start all NaN markers
    column_density = np.full([np.sum(ix_dla_range)], np.nan, dtype=np.float32)
    # Add DLAs, this loop will work from the DLA outward updating the offset values and not update it
    # if it would overwrite something set by another nearby DLA
    for i in range(int(samplerangepx+1)):
        for ix_dla,j in zip(ix_dlas,range(len(ix_dlas))):
            offsets_array[ix_dla+i] = -i if np.isnan(offsets_array[ix_dla+i]) else offsets_array[ix_dla+i]
            offsets_array[ix_dla-i] =  i if np.isnan(offsets_array[ix_dla-i]) else offsets_array[ix_dla-i]
            column_density[ix_dla+i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla+i]) else column_density[ix_dla+i]
            column_density[ix_dla-i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla-i]) else column_density[ix_dla-i]
    offsets_array = np.nan_to_num(offsets_array)
    column_density = np.nan_to_num(column_density)

    # Append these to the Sightline
    sightline.classification = classification
    sightline.offsets = offsets_array
    sightline.column_density = column_density

    # classification is 1 / 0 / -1 for DLA/nonDLA/border
    # offsets_array is offset
    return classification, offsets_array, column_density

def select_samples_50p_pos_neg(sightline):
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
    num_pos = np.sum(sightline.classification==1, dtype=np.float64)
    num_neg = np.sum(sightline.classification==0, dtype=np.float64)
    n_samples = int(min(num_pos, num_neg))

    r = np.random.permutation(len(sightline.classification))

    pos_ixs = r[sightline.classification[r]==1][0:n_samples]
    neg_ixs = r[sightline.classification[r]==0][0:n_samples]
    return np.hstack((pos_ixs,neg_ixs))
