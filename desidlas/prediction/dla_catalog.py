#these modules are used to generate QSO catalog and real DLA catalog (for mocks)
from astropy.table import Table, vstack
from astropy.io import fits
import numpy as np
def generate_qso_table(sightlines):
    """
    generate a QSO table for fitting BAO.
    
    Parameters
    ----------------
    sightlines: list of 'dla_cnn.data_model.Sightline.Sightline` object
    
    Return
    ----------------
    qso_tbl: astropy.table.Table object
    
    """
    qso_tbl = Table(names=('Plate','FiberID','MJD','TARGET_RA','TARGET_DEC', 'ZQSO','TARGETID','S/N'),dtype=('int','int','int','float','float','float','int','float'),meta={'EXTNAME': 'QSOCAT'})
    for ii in range(0,len(sightlines)):
        sightline=sightlines[ii]
        qso_tbl.add_row((sightline.id,sightline.id,sightline.id,sightline.ra,sightline.dec,sightline.z_qso,sightline.id,sightline.s2n))
    return qso_tbl
   
def generate_real_table(sightlines):
    """
    generate a real DLA&sub-DLA table.
    Parameters
    ----------------
    sightlines: dla_cnn.data_model.Sightline object list
    
    Return
    ----------------
    real_dla_tbl: astropy.table.Table object
    """
    real_dla_tbl = Table(names=('TARGET_RA','TARGET_DEC', 'ZQSO','Z','TARGETID','S/N','DLAID','NHI','DLA_CONFIDENCE','NHI_STD','ABSORBER_TYPE'),dtype=('float','float','float','float','int','float','str','float','float','float','str'),meta={'EXTNAME': 'DLACAT'})
    for ii in range(0,len(sightlines)):
        sightline=sightlines[ii]
        for dla in sightline.dlas:
            absorber_type =  "DLA" if dla.col_density >= 20.3 else "SUBDLA"
            real_dla_tbl.add_row((sightline.ra,sightline.dec,sightline.z_qso,float(dla.central_wavelength/1215.67-1),
                                  sightline.id,sightline.s2n,
                                  str(sightline.id)+dla.id,float(dla.col_density),1.0,0.0,absorber_type))
    return real_dla_tbl
    
def catalog_fits(sightlines,dlafile=None,qsofile=None):
    """
    save real DLA and QSO catalog.
    
    Parameters
    ----------------
    sightlines: list of 'dla_cnn.data_model.Sightline.Sightline` object
    dlafile: str
    qsofile: str
    
    Return
    ----------------
    real_dla_tbl: astropy.table.Table object
    
    """
    real_dla_tbl=generate_real_table(sightlines)
    qso_tbl=generate_qso_table(sightlines)
    real_dla_tbl.write(dlafile,overwrite=True)
    qso_tbl.write(qsofile,overwrite=True)
    return real_dla_tbl
