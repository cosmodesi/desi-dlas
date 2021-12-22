from astropy.io import fits
import numpy as np
from desidlas.data_model.Sightline import Sightline
from desidlas.data_model.Dla import Dla
from desidlas.datasets import preprocess 
from desidlas.dla_cnn.defs import best_v

class DesiMock: 
    """
    a class to load all spectrum from a mock DESI data v9 fits file, each file contains about 1186 spectrum.
    ------------------------------------------------------------------------------
    attributes:
    wavelength: array-like, the wavelength of all spectrum (all spectrum share same wavelength array)
    data: dict, using each spectra's id as its key and a dict of all data we need of this spectra as its value
      its format like spectra_id: {'FLUX':flux,'ERROR':error,'z_qso':z_qso, 'RA':ra, 'DEC':dec, 'DLAS':a tuple of Dla objects containing the information of dla}
    split_point_br: int,the length of the flux_b,the split point of  b channel data and r channel data
    split_point_rz: int,the length of the flux_b and flux_r, the split point of  r channel data and z channel data
    data_size: int,the point number of all data points of wavelength and flux
    """

    def _init_(self, wavelength = None, data = {}, split_point_br = None, split_point_rz = None, data_size = None):
        self.wavelength = wavelength
        self.data = data
        self.split_point_br = split_point_br
        self.split_point_rz = split_point_rz
        self.data_size = data_size

    def read_fits_file(self, spec_path, truth_path, zbest_path):
        """
        read Desi Mock spectrum from a fits file, load all spectrum as a DesiMock object
        ------------------------------------------------------------------------------------------------
        parameters:
        
        spec_path:  str, spectrum file path
        truth_path: str, truth file path
        zbest_path: str, zbest file path
        ------------------------------------------------------------------------------------------------
        return:
        
        self.wavelength,self.data(contained all information we need),self.split_point_br,self.split_point_rz,self.data_size
        """
        spec = fits.open(spec_path)
        zbest = fits.open(zbest_path)
        
        # spec[2].data ,spec[7].data and spec[12].data are the wavelength data for the b, r and z cameras.
        self.wavelength = np.hstack((spec['B_WAVELENGTH'].data.copy(), spec['R_WAVELENGTH'].data.copy(), spec['Z_WAVELENGTH'].data.copy()))
        self.data_size = len(self.wavelength)
        if truth_path !=[]:
            truth = fits.open(truth_path)
            if truth[3].data ==None:
                spec_dlas={}
            else:
                dlas_data = truth[3].data[truth[3].data.copy()['NHI']>19.3]
                spec_dlas = {}
        # item[2] is the spec_id, item[3] is the dla_id, and item[0] is NHI, item[1] is z_qso
                for item in dlas_data:
                    if item[2] not in spec_dlas:
                        spec_dlas[item[2]] = [Dla((item[1]+1)*1215.6701, item[0], '00'+str(item[3]-item[2]*1000))]
                    else:
                        spec_dlas[item[2]].append(Dla((item[1]+1)*1215.6701, item[0], '00'+str(item[3]-item[2]*1000)))

                test = np.array([True if item in dlas_data['TARGETID'] else False for item in spec[1].data['TARGETID'].copy()])
                for item in spec[1].data['TARGETID'].copy()[~test]:
                    spec_dlas[item] = []
        else:
            truth=[]
            spec_dlas={}
        

        # read data from the fits file above, one can directly get those varibles meanings by their names.
        spec_id = spec[1].data['TARGETID'].copy()
        flux_b = spec['B_FLUX'].data.copy()
        flux_r = spec['R_FLUX'].data.copy()
        flux_z = spec['Z_FLUX'].data.copy()
        flux = np.hstack((flux_b,flux_r,flux_z))
        ivar_b = spec['B_IVAR'].data.copy()
        ivar_r = spec['R_IVAR'].data.copy()
        ivar_z = spec['Z_IVAR'].data.copy()
        error = 1./np.sqrt(np.hstack((ivar_b,ivar_r,ivar_z)))
        pixel_mask=np.hstack((spec['B_MASK'].data.copy(),spec['R_MASK'].data.copy(),spec['Z_MASK'].data.copy()))
        self.split_point_br = flux_b.shape[1]
        self.split_point_rz = flux_b.shape[1]+flux_r.shape[1]
        z_qso = zbest[1].data['Z'].copy()
        zwarn=zbest[1].data['ZWARN'].copy()
        spectype=zbest[1].data['SPECTYPE'].copy()
        objtype=spec[1].data['OBJTYPE'].copy()
        ra = spec[1].data['TARGET_RA'].copy()
        dec = spec[1].data['TARGET_DEC'].copy()
        #add WISE data
        w1=spec[1].data['FLUX_W1'].copy()
        w2=spec[1].data['FLUX_W2'].copy()
        try:
            w1_ivar=spec[1].data['FLUX_IVAR_W1'].copy()
        except:
            w1_ivar=[0]*len(spec_id)
    
        try:
            w2_ivar=spec[1].data['FLUX_IVAR_W2'].copy()
        except:
            w2_ivar=[0]*len(spec_id)
        
        if spec_dlas !={}:
            self.data = {spec_id[i]:{'FLUX':flux[i],'ERROR': error[i], 'z_qso':z_qso[i] , 'RA': ra[i], 'DEC':dec[i],'spectype':spectype[i],'objtype':objtype[i], 'ZWARN':zwarn[i],'MASK':pixel_mask[i],'W1':w1[i],'W2':w2[i],'W1_IVAR':w1_ivar[i],'W2_IVAR':w2_ivar[i],'DLAS':spec_dlas[spec_id[i]]} for i in range(len(spec_id))}
        else:
            self.data = {spec_id[i]:{'FLUX':flux[i],'ERROR': error[i], 'z_qso':z_qso[i] , 'RA': ra[i], 'DEC':dec[i], 'spectype':spectype[i],'objtype':objtype[i],'ZWARN':zwarn[i],'MASK':pixel_mask[i],'W1':w1[i],'W2':w2[i],'W1_IVAR':w1_ivar[i],'W2_IVAR':w2_ivar[i],'DLAS':[]} for i in range(len(spec_id))}

    def get_sightline(self, id, camera = 'all', rebin=False, normalize=False):
        """
        using id(int) as index to retrive each spectra in DesiMock's dataset, return  a Sightline object.
        ---------------------------------------------------------------------------------------------------
        parameters:
        id: spectra's id , a unique number for each spectra.
        camera: str, 'b' : Load up the wavelength and data for the blue camera., 'r': Load up the wavelength and data for the r camera,
                     'z' : Load up the wavelength and data for the z camera, 'all':  Load up the wavelength and data for all cameras.
        rebin: bool, if True rebin the spectra to the best dlambda/lambda, default False,
        normalize: bool, if True normalize the spectra, using the slice of flux from wavelength ~1070 to 1170, default False.
        ---------------------------------------------------------------------------------------------------
        return:
        sightline: dla_cnn.data_model.Sightline.Sightline object
        """
        assert camera in ['all', 'r', 'z', 'b'], "No such camera! The parameter 'camera' must be in ['all', 'r', 'b', 'z']"
        sightline = Sightline(id)

        # this inside method can get the data(wavelength, flux, error) from the start_point(int) to end_point(int)
        def get_data(start_point=0, end_point=self.data_size):
            """
            
            Parameters
            ----------
            start_point: int, the start index of the slice of the data(wavelength, flux, error), default 0
            end_point: int, the end index of the slice of the data(wavelength, flux, error), default the length of the data array
            
            Returns
            -------
            
            """
            
            sightline.flux = self.data[id]['FLUX'][start_point:end_point]
            sightline.error = self.data[id]['ERROR'][start_point:end_point]
            sightline.z_qso = self.data[id]['z_qso']
            sightline.ra = self.data[id]['RA']
            sightline.dec = self.data[id]['DEC']
            sightline.spectype=self.data[id]['spectype']
            sightline.objtype=self.data[id]['objtype']
            sightline.zwarn=self.data[id]['ZWARN']
            sightline.w1=self.data[id]['W1']
            sightline.w2=self.data[id]['W2']
            sightline.w1_ivar=self.data[id]['W1_IVAR']
            sightline.w2_ivar=self.data[id]['W2_IVAR']
            sightline.dlas = self.data[id]['DLAS']
            sightline.pixel_mask=self.data[id]['MASK']
            sightline.loglam = np.log10(self.wavelength[start_point:end_point])
            sightline.split_point_br = self.split_point_br
            sightline.split_point_rz = self.split_point_rz
            #modify error to exclude inf:
            sightline.error[np.nonzero(sightline.pixel_mask)]=0
            sightline.error[np.where(sightline.flux==0)]=0
        
        # invoke the inside function above to select different camera's data.
        if camera == 'all':
            get_data()
            # this part is to deal with the overlapping part(only blue channel and the red channel)
            blfbound = np.argwhere(abs(self.wavelength - self.wavelength[self.split_point_br])<0.000001)[0][0]
            rrgbound = np.argwhere(abs(self.wavelength - self.wavelength[self.split_point_br-1])<0.000001)[1][0]
            #rzfbound = np.argwhere(abs(self.wavelength - self.wavelength[self.split_point_rz])<0.000001)[0][0]
            #rzgbound = np.argwhere(abs(self.wavelength - self.wavelength[self.split_point_rz-1])<0.000001)[1][0]
            rzfbound = np.argwhere(abs(self.wavelength - self.wavelength[self.split_point_rz])<0.4004)[0][0]
            rzgbound = np.argwhere(abs(self.wavelength - self.wavelength[self.split_point_rz-1])<0.4)[1][0]
            overlap_flux_b = sightline.flux[blfbound:self.split_point_br]
            overlap_flux_r = sightline.flux[self.split_point_br:rrgbound+1]
            overlap_error_b = sightline.error[blfbound:self.split_point_br]
            overlap_error_r = sightline.error[self.split_point_br:rrgbound+1]
            overlap_mask_b = sightline.pixel_mask[blfbound:self.split_point_br]
            overlap_mask_r = sightline.pixel_mask[self.split_point_br:rrgbound+1]
            
            overlap_flux_rr = sightline.flux[rzfbound:self.split_point_rz]
            overlap_flux_z = sightline.flux[self.split_point_rz:rzgbound+1]
            overlap_error_rr = sightline.error[rzfbound:self.split_point_rz]
            overlap_error_z = sightline.error[self.split_point_rz:rzgbound+1]
            overlap_mask_rr = sightline.pixel_mask[rzfbound:self.split_point_rz]
            overlap_mask_z = sightline.pixel_mask[self.split_point_rz:rzgbound+1]
            
            test_1 = overlap_error_b<overlap_error_r
            overlap_flux_1 = np.array([overlap_flux_b[i] if test_1[i] else overlap_flux_r[i] for i in range(-blfbound+self.split_point_br)])
            overlap_error_1 = np.array([overlap_error_b[i] if test_1[i] else overlap_error_r[i] for i in range(-blfbound+self.split_point_br)])
            overlap_mask_1 = np.array([overlap_mask_b[i] if test_1[i] else overlap_mask_r[i] for i in range(-blfbound+self.split_point_br)])
            
            test_2 = overlap_error_rr<overlap_error_z
            overlap_flux_2 = np.array([overlap_flux_rr[i] if test_2[i] else overlap_flux_z[i] for i in range(-rzfbound+self.split_point_rz)])
            overlap_error_2 = np.array([overlap_error_rr[i] if test_2[i] else overlap_error_z[i] for i in range(-rzfbound+self.split_point_rz)])
            overlap_mask_2 = np.array([overlap_mask_rr[i] if test_2[i] else overlap_mask_z[i] for i in range(-rzfbound+self.split_point_rz)])
            
            sightline.flux = np.hstack((sightline.flux[:blfbound],overlap_flux_1,sightline.flux[rrgbound+1:rzfbound],overlap_flux_2,sightline.flux[rzgbound+1:]))
            sightline.error = np.hstack((sightline.error[:blfbound],overlap_error_1,sightline.error[rrgbound+1:rzfbound],overlap_error_2,sightline.error[rzgbound+1:]))
            sightline.pixel_mask = np.hstack((sightline.pixel_mask[:blfbound],overlap_mask_1,sightline.pixel_mask[rrgbound+1:rzfbound],overlap_mask_2,sightline.pixel_mask[rzgbound+1:]))
            sightline.loglam = np.hstack((sightline.loglam[:self.split_point_br],sightline.loglam[rrgbound+1:self.split_point_rz],sightline.loglam[rzgbound+1:]))
            sightline.split_point_rz=sightline.split_point_rz-len(overlap_flux_b)
        elif camera == 'b':
            get_data(end_point = self.split_point_br)
        elif camera == 'r':
            get_data(start_point= self.split_point_br, end_point= self.split_point_rz)
        else:
            get_data(start_point=self.split_point_rz)

        # if the parameter rebin is True, then rebin this sightline using rebin method in preprocess.py and the v we determined previously(defs.py/best_v) .
        if rebin:
            preprocess.rebin(sightline, best_v[camera])
        #if the parameter normalize is True, then normalize this sightline using the method in preprocess.py
        if normalize:
            preprocess.normalize(sightline, self.wavelength, self.data[id]['FLUX'])
        sightline.s2n = preprocess.estimate_s2n(sightline)
        # Return the Sightline object
        return sightline
