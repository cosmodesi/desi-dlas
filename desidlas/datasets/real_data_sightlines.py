"""Catalog-driven sightline generation for real DESI spectra.

This module preserves the real-data preprocessing flow used by the older
notebook/script pipeline: select targets from a QSO catalog, read coadd spectra
without zbest/truth, assign redshift metadata from the catalog, then run the
same normalize/rebin/cut_z_region preprocessing.
"""

import os
import time

import numpy as np
import scipy.io as scio
from astropy.table import Table
from tqdm import tqdm

from desidlas.datasets import preprocess
from desidlas.datasets.DesiMock import DesiMock


def healpix_group(pixel):
    group = str(pixel)[:-2]
    return group if group else "0"


def prepare_qso_catalog(qso_catalog, z_min=2.1, z_max=5.8):
    """Read a QSO catalog and add the same selection columns as the old script."""
    qsos = Table.read(qso_catalog)
    qsos["z_ind"] = (
        (qsos["Z"] > z_min) & (qsos["ZWARN"] == 0) & (qsos["Z"] < z_max)
    ).astype(int)

    if "BI_CIV" in qsos.colnames and "AI_CIV" in qsos.colnames:
        bal_ind = (qsos["BI_CIV"] == 0) & (qsos["AI_CIV"] < 500)
    elif "bal_ind" in qsos.colnames:
        bal_ind = qsos["bal_ind"].astype(bool)
    else:
        bal_ind = np.ones(len(qsos), dtype=bool)

    qsos["bal_ind"] = bal_ind.astype(int)
    qsos["ind"] = (bal_ind & (qsos["z_ind"] == 1)).astype(int)
    return qsos


def select_qsos_for_pixel(qsos, pixel, use_bal=False):
    pix_qsos = qsos[qsos["HPXPIXEL"] == int(pixel)]
    if use_bal:
        return pix_qsos[pix_qsos["ind"] == 1]
    return pix_qsos[pix_qsos["z_ind"] == 1]


def preprocess_sightlines(sightlines, qsocat, v=44735, out_path=None):
    """Preprocess real-data sightlines exactly as the old data script did."""
    pre_sightlines = []
    qsocat["S2N"] = 0.0
    qsocat["process_ind"] = 0
    qsocat["z_start"] = 0.0
    qsocat["z_end"] = 0.0
    qsocat["z_use"] = 0

    for i in tqdm(range(len(sightlines))):
        sightline = sightlines[i]
        if sightline != []:
            assert sightline.id == qsocat[i]["TARGETID"]
            sightline.s2n = preprocess.estimate_s2n(sightline)
            if sightline.s2n > 0:
                preprocess.normalize(sightline, 10**sightline.loglam, sightline.flux)
                preprocess.rebin(sightline, v)
                z_start, z_end, z_use = preprocess.cut_z_region(sightline)
                pre_sightlines.append(sightline)
                qsocat[i]["S2N"] = sightline.s2n
                qsocat[i]["process_ind"] = 1
                qsocat[i]["z_start"] = z_start
                qsocat[i]["z_end"] = z_end
                qsocat[i]["z_use"] = z_use
            else:
                pre_sightlines.append([])
                qsocat[i]["S2N"] = sightline.s2n
                qsocat[i]["process_ind"] = 0
        else:
            pre_sightlines.append([])

    if out_path:
        np.save(out_path, pre_sightlines)

    return pre_sightlines, qsocat


def preprocess_forgp(sightlines, qsocat, output_path=None):
    """Write the GP preload MAT file produced by the old real-data pipeline."""
    all_wavelengths = []
    all_flux = []
    all_noise_variance = []
    sightline_ids = []
    loading_min_lambda = 910
    loading_max_lambda = 1217
    qsocat["process_gp_ind"] = 0

    for i in tqdm(range(len(sightlines))):
        sightline = sightlines[i]
        if (qsocat[i]["process_ind"] == 1) & (sightline != []):
            assert sightline.id == qsocat[i]["TARGETID"]
            this_wavelength = 10**sightline.loglam
            flux = sightline.flux
            var = sightline.error**2
            rest_wavelength = this_wavelength / (1 + sightline.z_qso)
            sightline_ids.append(sightline.id)
            ind = (rest_wavelength >= loading_min_lambda) & (rest_wavelength <= loading_max_lambda)
            if sum(ind) != 0:
                ind[max(0, np.nonzero(ind)[0][0] - 1)] = True
                ind[min(np.nonzero(ind)[0][-1] + 1, len(np.nonzero(ind)[0] - 1))] = True
                all_wavelengths.append(list(this_wavelength[ind]))
                all_flux.append(list(flux[ind]))
                all_noise_variance.append(list(var[ind]))
                qsocat[i]["process_gp_ind"] = 1
            else:
                sightline_ids.append([])
                all_wavelengths.append([])
                all_flux.append([])
                all_noise_variance.append([])
                print(sightline.id)
                qsocat[i]["process_gp_ind"] = 0
        else:
            sightline_ids.append([])
            all_wavelengths.append([])
            all_flux.append([])
            all_noise_variance.append([])
            qsocat[i]["process_gp_ind"] = 0

    if output_path:
        scio.savemat(
            output_path,
            {
                "sightline_ids": sightline_ids,
                "wavelengths": all_wavelengths,
                "flux": all_flux,
                "noise_variance": all_noise_variance,
            },
        )
    print("make preload_qsos done")
    return qsocat


def _catalog_column(qsocat, name, default=0):
    if name in qsocat.colnames:
        return list(qsocat[name])
    return [default] * len(qsocat)


def make_desi_data_sightlines(
    spectra_path,
    qsocat,
    output_dir,
    release,
    survey,
    program,
    pixel,
    v=44735,
    write_aux=True,
):
    """Generate one pixel's old-format real-data sightline outputs."""
    os.makedirs(output_dir, exist_ok=True)
    group = healpix_group(pixel)
    prefix = f"{release}-{survey}-{program}-{group}-{pixel}"

    raw_path = os.path.join(output_dir, f"{prefix}-raw-sightlines.npy")
    pre_path = os.path.join(output_dir, f"{prefix}-pre-sightlines.npy")
    premat_path = os.path.join(output_dir, f"{prefix}-preload_qsos.mat")
    catmat_path = os.path.join(output_dir, f"{prefix}-catalog.mat")
    catfits_path = os.path.join(output_dir, f"{prefix}-catalog.fits")

    start_time = time.time()
    specs = DesiMock()
    specs.read_fits_file(spectra_path, [], [])
    print(f"{release} {survey} {program} {group} {pixel}, {len(qsocat)} spectra")

    sightlines = []
    missing_targetids = []
    for row in qsocat:
        targetid = row["TARGETID"]
        try:
            sightline = specs.get_sightline(targetid, camera="all", rebin=False, normalize=False)
        except KeyError:
            sightlines.append([])
            missing_targetids.append(targetid)
            continue
        assert sightline.id == targetid
        sightline.z_qso = float(row["Z"])
        sightline.spectype = str(row["SPECTYPE"])
        sightline.zwarn = int(row["ZWARN"])
        sightlines.append(sightline)

    assert len(sightlines) == len(qsocat)
    if missing_targetids:
        print(
            f"Missing {len(missing_targetids)} of {len(qsocat)} catalog TARGETIDs in coadd "
            f"for pixel {pixel}; first missing TARGETID={missing_targetids[0]}"
        )
    print(f"Extracting sightlines time: {time.time() - start_time} seconds")
    np.save(raw_path, sightlines)

    process_qsocat = qsocat["TARGETID", "HPXPIXEL"]
    pre_sightlines, process_qsocat = preprocess_sightlines(
        sightlines, process_qsocat, v=v, out_path=pre_path
    )

    if write_aux:
        process_qsocat = preprocess_forgp(pre_sightlines, process_qsocat, output_path=premat_path)
        process_qsocat.write(catfits_path, overwrite=True)
        scio.savemat(
            catmat_path,
            {
                "ras": _catalog_column(qsocat, "TARGET_RA"),
                "decs": _catalog_column(qsocat, "TARGET_DEC"),
                "target_ids": _catalog_column(qsocat, "TARGETID"),
                "z_qsos": _catalog_column(qsocat, "Z"),
                "snrs": list(process_qsocat["S2N"]),
                "bal_visual_flags": list(np.ones(len(qsocat)) - qsocat["bal_ind"]),
                "filter_flags": list(np.ones(len(process_qsocat)) - process_qsocat["process_gp_ind"]),
                "zwarning": _catalog_column(qsocat, "ZWARN"),
            },
        )
        print("save qso catalogs done")

    return pre_path
