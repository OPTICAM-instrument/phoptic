from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable


from astropy.io import fits
import numpy as np
from numpy.typing import NDArray


from phoptic.instruments import Instrument, OPTICAM_MX
from phoptic.mef_slice import create_file_paths, MEFSlice
from phoptic.utils.helpers import camera_and_filter_key, camera_key
from phoptic.utils.image_helpers import rebin_image
from phoptic.utils.logging import log_file




class Corrector(ABC):
    """
    Base class for correctors.
    """


    def __init__(
        self,
        out_directory: Path | str | None = None,
        data_directory: Path | str | None = None,
        instrument: Instrument = OPTICAM_MX(),
        rebin_factor: int = 1,
        image_filter: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        *args,
        **kwargs,
        ) -> None:
        """
        Initialise the corrector.
        
        Parameters
        ----------
        out_directory : Path | str | None, optional
            The path to the output directory, by default `None`. Master correction images will be saved here.
        data_directory : Path | str | None, optional
            The path to the data directory, by default `None`. This must point to a directory containing a series of
            (multi-extension) FITS files.
        instrument : Instrument, optional
            The instrument, by default `OPTICAM_MX()`.
        rebin_factor : int, optional
            The factor by which to rebin the data, by default 1. Useful if, for example, calibration images were taken
            in a lower binning mode than the observations.
        image_filter : Callable[[NDArray[np.float64]], NDArray[np.float64]] | None, optional
            The kernel/filter to apply to calibration images as they are opened. Paez+2026:
            https://ui.adsabs.harvard.edu/abs/2026RASTI...5ag021P/abstract found that a 3x3 median filter
            (e.g., `scipy.ndimage.median_filter()`) can be used to correct for warm pixels in long exposures (> 10 s)
            with OPTICAM.
        """
        
        self.out_directory = Path(out_directory) if out_directory is not None else None
        self.instrument = instrument
        self.passed_checks = False
        
        assert isinstance(rebin_factor, int), "[PHOPTIC] Non-integer rebin factors are not supported!"
        self.rebin_factor = rebin_factor
        self.image_filter = image_filter
        
        if self.out_directory is not None:
            if not self.out_directory.is_dir():
                self.out_directory.mkdir(parents=True)
        
        # get the calibration image files
        if data_directory is not None:
            raw_data_files = create_file_paths(data_directory=Path(data_directory))
            assert len(raw_data_files) > 0, f'[PHOPTIC] No FITS files found in {Path(data_directory).resolve()}.'
            self.data_files = self._validate_data(raw_data_files)
        else:
            self.data_files = None
        
        self.master_images: dict[str, NDArray[np.float64]] = {}
        self.master_variances: dict[str, NDArray[np.float64]] = {}
        
        # load master images if they already exist
        if self.master_image_path is not None:
            if self.master_image_path.is_file():
                self._read_master_image()


    @property
    @abstractmethod
    def master_image_path(self) -> Path | None:
        """
        The path to the master calibration image.
        
        Returns
        -------
        Path | None
            The path to the master calibration image.
        """
        pass


    @abstractmethod
    def correct(
        self,
        image: NDArray[np.float64],
        *args,
        **kwargs,
        ) -> tuple[NDArray[np.float64], float | NDArray[np.float64]]:
        """
        Apply the required correction to an image.
        
        Parameters
        ----------
        image : NDArray[np.float64]
            The image.
        
        Returns
        -------
        tuple[NDArray[np.float64], float | NDArray[np.float64]]
            The corrected image and the variance of the correction term. The variance may be a `float` (e.g., if
            the dark noise is calculated from the exposure-integrated dark current) or an `NDArray`.
        """
        
        pass


    @abstractmethod
    def create_master_images(
        self,
        overwrite: bool = False,
        *args,
        **kwargs,
        ) -> None:
        """
        Create the master calibration image(s).
        
        Parameters
        ----------
        bias_corrector : BiasCorrector | None, optional
            The bias corrector, by default `None` (no bias corrections).
        overwrite : bool, optional
            Whether to overwrite any existing master calibration images, by default `False`.
        """
        
        pass


    def _read_master_image(
        self,
        ) -> None:
        """
        Read the master images from the output directory.
        """
        
        with fits.open(self.master_image_path) as hdul:
            # skip primary HDU since it doesn't contain any data
            for hdu in hdul[1:]:
                # filter info is saved using instrument format so it's safe to use the filter keyword instead of the
                # instrument.get_filter() method
                fltr = hdu.header[self.instrument.filter_kw]
                name = hdu.header['EXTNAME']
                data = np.asarray(hdu.data, dtype=np.float64)
                if name == 'DATA':
                    self.master_images[fltr] = data
                elif name == 'VARIANCE':
                    self.master_variances[fltr] = data


    @abstractmethod
    def run_checks(
        self,
        data_files_by_key: dict[str, list[MEFSlice]],
        return_errors: bool = False,
        ) -> None | int:
        """
        Run a series of checks on the corrector to ensure that it is compatible with the data.
        
        Parameters
        ----------
        data_files_by_key : dict[str, list[MEFSlice]]
            The science image files grouped by camera:filter keys.
        return_errors : bool, optional
            Whether to return the number of errors raised, by default `False`.
        
        Returns
        -------
        None | int
            If `return_errors=True`, the number of errors raised is returned.
        """
        
        pass


    def _save_master_image(
        self,
        overwrite: bool,
        ) -> None:
        """
        Save the master images and their corresponding variances to a compressed FITS cube.
        
        Parameters
        ----------
        overwrite : bool
            Whether to overwrite an existing master images file.
        """
        
        if self.__class__.__name__ == 'BiasCorrector':
            comment = 'This FITS file contains master bias images and their corresponding variances for each filter.'
        elif self.__class__.__name__ == 'DarkNoiseCorrector':
            comment = 'This FITS file contains master dark images and their corresponding variances for each filter.'
        elif self.__class__.__name__ == 'FlatFieldCorrector':
            comment = 'This FITS file contains master flat-field images and their corresponding variances for each filter.'
        else:
            raise ValueError(f'[PHOPTIC] Unrecognised corrector: {self.__class__.__name__}.')
        
        hdr = fits.Header()
        hdr['COMMENT'] = comment
        empty_primary = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([empty_primary])
        
        for fltr in self.master_images:
            # master image
            hdr = fits.Header()
            # filter is already in instrument format so no need to use instrument.get_filter()
            hdr[self.instrument.filter_kw] = fltr
            hdu = fits.ImageHDU(
                data=self.master_images[fltr],
                header=hdr,
                name='DATA',
                )
            hdul.append(hdu)
            
            # master variance
            hdr = fits.Header()
            # filter is already in instrument format so no need to use instrument.get_filter()
            hdr[self.instrument.filter_kw] = fltr
            hdu = fits.ImageHDU(
                data=self.master_variances[fltr],
                header=hdr,
                name='VARIANCE',
                )
            hdul.append(hdu)
        
        if not self.master_image_path.is_file() or overwrite:
            hdul.writeto(self.master_image_path, overwrite=overwrite)


    @abstractmethod
    def _validate_data(
        self,
        files: list[MEFSlice],
        ) -> dict[str, list[MEFSlice]]:
        """
        Validate the input data.
        
        Parameters
        ----------
        files : list[MEFSlice]
            The input files.
        
        Returns
        -------
        dict[str, list[MEFSlice]]
            The file paths to the input data separated by filter.
        """
        
        pass




class BiasCorrector(Corrector):
    """
    Helper clsas for performing bias corrections.
    """


    @property
    def master_image_path(self) -> Path | None:
        """
        The path to the master calibration image.
        
        Returns
        -------
        Path
            The path to the master calibration image.
        """
        
        return self.out_directory.joinpath('master_bias.fits.gz') if self.out_directory is not None else None


    def correct(
        self,
        image: NDArray[np.float64],
        camera: str,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Subtract the bias from an image.
        
        Parameters
        ----------
        image : NDArray[np.float64]
            The image.
        camera : str
            The camera that took the image.
        
        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            The corrected image and the variance of the master bias image.
        
        Raises
        ------
        ValueError
            If no bias images were found with the given filter.
        """
        
        if camera not in self.data_files.keys():
            raise ValueError(f"[PHOPTIC] No bias images found for {camera} filter.")
        if camera not in self.master_images.keys() or self.master_images[camera] is None:
            print(f'[PHOPTIC] {camera} master bias image not found. Attempting to create.')
            self.create_master_images()
        
        return image - self.master_images[camera], self.master_variances[camera]


    def create_master_images(
        self,
        overwrite: bool = False,
        ) -> None:
        """
        Create master bias images for each filter.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite the existing master bias image, by default `False`.
        """
        
        if self.master_image_path.is_file() and not overwrite:
            print(f'[PHOPTIC] Master bias file already exists. To overwrite, set overwrite=True.')
            return
        
        for camera in self.data_files.keys():
            
            if len(self.data_files[camera]) == 1:
                raise Exception(f"[PHOPTIC] Only one {camera} bias image found. Master bias images cannot be created from a single image.")
            
            biases = []
            for bias_path in self.data_files[camera]:
                bias = bias_path.get_data()
                
                if self.image_filter is not None:
                    bias = self.image_filter(bias)
                
                if self.rebin_factor > 1:
                    bias = rebin_image(image=bias, factor=self.rebin_factor)
                
                biases.append(bias)
            
            # use mean since bias frames shouldn't contain outliers like cosmic rays
            self.master_images[camera] = np.mean(biases, axis=0)
            self.master_variances[camera] = np.var(biases, axis=0, ddof=1) / len(biases)
        
        print('[PHOPTIC] Master bias image(s) created.')
        
        self._save_master_image(overwrite=overwrite)
        
        print(f'[PHOPTIC] Master bias image(s) saved to {self.master_image_path}.')


    def run_checks(
        self,
        data_files_by_key: dict[str, list[MEFSlice]],
        return_errors: bool = False,
        ) -> None | int:
        """
        Run a series of checks on the corrector to ensure that it is compatible with the data. In this case, check the
        binning of the science images matches those of the bias images (ignoring self.rebin_factor) and that there are
        no missing filters.
        
        Parameters
        ----------
        data_files_by_key : dict[str, list[MEFSlice]
            The science image files grouped by camera:filter keys.
        return_errors : bool, optional
            Whether to return the number of errors raised, by default `False`.
        
        Returns
        -------
        None | int
            If `return_errors=True`, the number of errors raised is returned. Otherwise, nothing is returned.
        """
        
        bias_path = next(iter(self.data_files.values()))[0]  # get the path to a random flat
        bias_header = bias_path.get_header()
        image_path = next(iter(data_files_by_key.values()))[0]  # get the path to a science image
        science_header = image_path.get_header()
        
        science_keys = list(data_files_by_key.keys())
        science_cameras = [camera_key(science_key) for science_key in science_keys]
        bias_cameras = list(self.data_files.keys())
        
        errors = 0
        warnings = 0
        
        #################################################### errors ####################################################
        
        # check filters match
        if not all([science_camera in bias_cameras for science_camera in science_cameras]):
            errors += 1
            print(f'[PHOPTIC] ERROR: inconsistent cameras found between the bias images and the science images. Bias cameras: ({','.join(bias_cameras)}); science cameras: ({','.join(science_cameras)})')
        
        ################################################### warnings ###################################################
        
        # check binnings match
        bias_binning = self.instrument.get_binning(header=bias_header)
        science_binning = self.instrument.get_binning(header=science_header)
        if bias_binning != science_binning:
            warnings += 1
            print(f'[PHOPTIC] WARNING: inconsistent binning found between the bias images and the science images. Bias image binning: {bias_binning}; science image binning: {science_binning}. If you have passed a suitable rebin_factor to your BiasCorrector instance, then you can safely ignore this warning.')
        
        ################################################### summary ###################################################
        
        if errors > 0 or warnings > 0:
            print()  # blank line for readibility
        
        if errors == 0:
            print(f'[PHOPTIC] BiasCorrector sucessfully passed all checks.')
        else:
            if errors == 1:
                print('[PHOPTIC] BiasCorrector failed 1 check.')
            else:
                print(f'[PHOPTIC] BiasCorrector failed {errors} checks.')
        
        if warnings == 1:
            print('[PHOPTIC] BiasCorrector triggered a warning during 1 check. Warnings may be ignored provided their caveats are satisfied.')
        elif warnings > 1:
            print(f'[PHOPTIC] BiasCorrector triggered a warning during {warnings} checks. Warnings may be ignored provided their caveats are satisfied.')
        
        if errors == 0:
            self.passed_checks = True
        
        if return_errors:
            return errors


    def _validate_data(
        self,
        files: list[MEFSlice],
        ) -> dict[str, list[MEFSlice]]:
        """
        Ensure that the bias images in the specified directory are valid (i.e., all use the same binning and have 
        exposure times of 0 s).
        
        Parameters
        ----------
        files : list[MEFSlice]
            The bias image files.
        
        Returns
        -------
        dict[str, list[MEFSlice]]
            A dictionary containing the bias image files for each camera.
        """
        
        cameras: dict[str, str] = {}
        binnings: dict[str, str] = {}
        exptimes: dict[str, float] = {}
        validated_files: dict[str, list[MEFSlice]] = {}
        
        for file in files:
            header = file.get_header()
            camera = self.instrument.get_camera(header=header)
            
            cameras[file.key] = camera
            binnings[file.key] = self.instrument.get_binning(header=header)
            exptimes[file.key] = float(header[self.instrument.exptime_kw])
            
            if camera not in validated_files.keys():
                validated_files[camera] = []
            validated_files[camera].append(file)
        
        unique_binnings = set(binnings.values())
        if len(unique_binnings) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='binnings.json',
                file_contents=binnings,
                )
            raise ValueError(f'[PHOPTIC] Inconsistent binning detected in the bias images. Image binnings have been logged to {self.out_directory.joinpath('diag/binnings.json')}.')
        
        unique_exptimes = set(exptimes.values())
        if len(unique_exptimes) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='exptimes.json',
                file_contents=exptimes,
                )
            raise ValueError(f'[PHOPTIC] Invalid exposure times detected in the bias images. Exposure times have been logged to {self.out_directory.joinpath('diag/exptimes.json')}. All bias images should have an exposure time of 0.0 s.')
        
        for camera, valid_files in validated_files.items():
            print(f'[PHOPTIC] {len(valid_files)} {camera} bias images.')
        
        return validated_files




class DarkNoiseCorrector(Corrector):
    """
    Helper class for performing dark noise corrections.
    """


    def __init__(
        self,
        out_directory: Path | str | None = None,
        data_directory: Path | str | None = None,
        instrument: Instrument = OPTICAM_MX(),
        rebin_factor: int = 1,
        image_filter: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        bias_corrector: BiasCorrector | None = None,
        ) -> None:
        """
        Initialise the dark noise corrector.
        
        Parameters
        ----------
        out_directory : Path | str | None, optional
            The path to the output directory, by default `None`. Master dark images will be saved here.
        data_directory : Path | str | None, optional
            The path to the data directory, by default `None`. This must point to a directory containing a series of
            dark images.
        instrument : Instrument, optional
            The instrument, by default `OPTICAM_MX()`.
        rebin_factor : int, optional
            The factor by which to rebin the data, by default 1. Useful if, for example, calibration images were taken
            using a higher resolution than the science images.
        image_filter : Callable[[NDArray[np.float64]], NDArray[np.float64]] | None, optional
            The kernel/filter to apply to calibration images as they are opened. Paez+2026:
            https://ui.adsabs.harvard.edu/abs/2026RASTI...5ag021P/abstract found that a 3x3 median filter
            (e.g., `scipy.ndimage.median_filter()`) can be used to correct for warm pixels in long exposures (> 10 s)
            with OPTICAM.
        bias_corrector : BiasCorrector | None, optional
            The bias corrector to use to bias-correct the dark images, by default `None`. If `None`, no bias
            corrections are performed.
        """
        
        self.bias_corrector = bias_corrector
        
        super().__init__(
                         out_directory=out_directory,
                         data_directory=data_directory,
                         instrument=instrument,
                         rebin_factor=rebin_factor,
                         image_filter=image_filter,
                         )


    @property
    def master_image_path(self) -> Path | None:
        """
        The path to the master calibration image.
        
        Returns
        -------
        Path
            The path to the master calibration image.
        """
        
        return self.out_directory.joinpath('master_darks.fits.gz') if self.out_directory is not None else None


    def correct(
        self,
        image: NDArray[np.float64],
        camera: str | None = None,
        fltr: str | None = None,
        key: str | None = None,
        dark_flux: float | None = None,
        ) -> tuple[NDArray[np.float64], float | NDArray[np.float64]]:
        """
        Subtract the dark noise from an image.
        
        Parameters
        ----------
        image : NDArray[np.float64]
            The image.
        camera : str
            The camera that took the image.
        fltr : str
            The image filter.
        dark_flux : float | None, optional
            The exposure-integrated dark current, by default `None`. If the instrument provides a measure of the dark
            current in the image header, this obviates the need for master darks.
        
        Returns
        -------
        NDArray[np.float64] | tuple[NDArray[np.float64], float]
            The corrected image and the variance of the master dark image.
        
        Raises
        ------
        ValueError
            If no dark images were found with the given filter.
        """
        
        if key is None:
            key = camera_and_filter_key(camera, fltr)
        
        if dark_flux is None:
            if key not in self.data_files.keys():
                raise ValueError(f"[PHOPTIC] No dark images found for {key}.")
            if key not in self.master_images.keys() or self.master_images[key] is None:
                print(f'[PHOPTIC] {key} master dark image not found. Attempting to create.')
                self.create_master_images()
            
            return image - self.master_images[key], self.master_variances[key]
        else:
            return image - dark_flux, dark_flux


    def create_master_images(
        self,
        overwrite: bool = False,
        ) -> None:
        """
        Create master dark images for each available filter.
        
        Parameters
        ----------
        overwrite : bool, optional
            Whether to overwrite any existing master dark image, by default `False`.
        """
        
        if self.master_image_path.is_file() and not overwrite:
            print(f'[PHOPTIC] Master darks file already exists. To overwrite existing master darks, set overwrite=True.')
            return
        
        for key in self.data_files.keys():
            
            if len(self.data_files[key]) == 1:
                raise Exception(f"[PHOPTIC] Only one {key} dark image found. Master darks cannot be created from a single image.")
            
            # read darks
            darks = []
            for dark_file in self.data_files[key]:
                dark = dark_file.get_data()
                
                if self.image_filter is not None:
                    dark = self.image_filter(dark)
                
                if self.rebin_factor > 1:
                    dark = rebin_image(image=dark, factor=self.rebin_factor, )
                
                # apply bias correction
                if self.bias_corrector is not None:
                    dark, bias_var = self.bias_corrector.correct(
                        image=dark,
                        camera=camera_key(key),
                        )
                else:
                    bias_var = 0.
                
                darks.append(dark)
            
            self.master_images[key] = np.median(darks, axis=0)
            self.master_variances[key] = np.pi / (2 * len(darks)) * (np.var(darks, axis=0, ddof=1) + bias_var)
        
        print('[PHOPTIC] Master dark image(s) created.')
        
        self._save_master_image(overwrite=overwrite)
        
        print(f'[PHOPTIC] Master dark image(s) saved to {self.master_image_path}.')


    def run_checks(
        self,
        data_files_by_key: dict[str, list[MEFSlice]],
        return_errors: bool = False,
        ) -> None | int:
        """
        Run a series of checks on the corrector to ensure that it is compatible with the data. In this case, check the
        binning of the science images matches to those of the darks (neglecting self.rebin_factor), there are no missing
        filters, and the exposure times of the science images matches those of the darks.
        
        Parameters
        ----------
        data_files_by_key : dict[str, list[MEFSlice]
            The science image files grouped by camera:filter keys.
        return_errors : bool, optional
            Whether to return the number of errors raised, by default `False`.
        
        Returns
        -------
        None | int
            If `return_errors=True`, the number of errors raised is returned. Otherwise, nothing is returned.
        """
        
        # get some random image headers from the dark images and the science images
        if self.data_files is not None:
            dark_file = next(iter(self.data_files.values()))[0]  # path to a random flat
            dark_header = dark_file.get_header()
            dark_binning = self.instrument.get_binning(header=dark_header)
        image_file = next(iter(data_files_by_key.values()))[0]  # path to a science image
        science_header = image_file.get_header()
        science_binning = self.instrument.get_binning(header=science_header)
        
        errors = 0
        warnings = 0
        
        #################################################### errors ####################################################
        
        # check exposure times match
        if self.data_files is not None:
            dark_exptime = dark_header[self.instrument.exptime_kw]
            science_exptime = science_header[self.instrument.exptime_kw]
            if dark_exptime != science_exptime:
                errors += 1
                print(f'[PHOPTIC] ERROR: inconsistent exposure times found between the dark images and the science images. Dark image exposure time: {dark_exptime}; science image exposure time: {science_exptime}')
        
        # check filters match
        if self.data_files is not None:
            if self.data_files.keys() != data_files_by_key.keys():
                errors += 1
                print(f'[PHOPTIC] ERROR: inconsistent filters found between the dark images and the science images. Dark image filters: ({','.join(self.data_files.keys())}); science image filters: ({','.join(data_files_by_key.keys())})')
        
        if self.data_files is None:
            try:
                float(self.instrument.get_dark_flux(header=science_header))
            except Exception as e:
                errors += 1
                print(f'[PHOPTIC] ERROR: No dark images passed to DarkCurrentCorrector and the dark noise could not be inferred from file: {image_file.path} extension {image_file.ext} due to the exception {e} This may be due to an incorrect dark current keyword or the images not including a dark current keyword in their headers. In the latter case, dedicated dark images will be required to quantify the dark noise.')
        
        ################################################### warnings ###################################################
        
        # check binnings match
        if self.data_files is not None:
            if dark_binning != science_binning:
                warnings += 1
                print(f'[PHOPTIC] WARNING: inconsistent binning found between the dark images and the science images. Dark image binning: {dark_binning}; science image binning: {science_binning}. If you have passed a suitable rebin_factor to your DarkNoiseCorrector instance, then you can safely ignore this warning.')
        
        ################################################### summary ###################################################
        
        if errors > 0 or warnings > 0:
            print()  # blank line for readibility
        
        if errors == 0:
            print(f'[PHOPTIC] DarkNoiseCorrector sucessfully passed all checks.')
        else:
            if errors == 1:
                print('[PHOPTIC] DarkNoiseCorrector failed 1 check.')
            else:
                print(f'[PHOPTIC] DarkNoiseCorrector failed {errors} checks.')
        
        if warnings == 1:
            print('[PHOPTIC] DarkNoiseCorrector triggered a warning during 1 check. Warnings may be ignored provided their caveats are satisfied.')
        elif warnings > 1:
            print(f'[PHOPTIC] DarkNoiseCorrector triggered a warning during {warnings} checks. Warnings may be ignored provided their caveats are satisfied.')
        
        if errors == 0:
            self.passed_checks = True
        
        if return_errors:
            return errors


    def _validate_data(
        self,
        files: list[MEFSlice],
        ) -> dict[str, list[MEFSlice]]:
        """
        Ensure that the dark images in the specified directory are valid (i.e., all use the same binning).
        
        Parameters
        ----------
        file_paths : list[MEFSlice]
            The dark image files
        
        Returns
        -------
        dict[str, list[Path]]
            A dictionary containing the dark image files for each filter.
        """
        
        keys: dict[str, str] = {}
        binnings: dict[str, str] = {}
        exptimes: dict[str, float] = {}
        validated_files: dict[str, list[MEFSlice]] = {}
        
        for file in files:
            header = file.get_header()
            
            fltr = self.instrument.get_filter(header=header)
            camera = self.instrument.get_camera(header=header)
            key = camera_and_filter_key(camera, fltr)
            
            keys[file.key] = key
            binnings[file.key] = self.instrument.get_binning(header=header)
            exptimes[file.key] = float(header[self.instrument.exptime_kw])
            
            if key not in validated_files.keys():
                validated_files[key] = []
            validated_files[key].append(file)
        
        unique_binnings = set(binnings.values())
        
        if len(unique_binnings) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='binnings.json',
                file_contents=binnings,
                )
            raise ValueError(f'[PHOPTIC] Inconsistent binning detected in the dark images. Image binnings have been logged to {self.out_directory.joinpath('diag/binnings.json')}')
        
        unique_exptimes = set(exptimes.values())
        if len(unique_exptimes) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='exptimes.json',
                file_contents=exptimes,
                )
            raise ValueError(f'[PHOPTIC] Inconsistent exposure times detected in the dark images. Exposure times have been logged to {self.out_directory.joinpath('diag/exptimes.json')}')
        
        for key, valid_files in validated_files.items():
            print(f'[PHOPTIC] {len(valid_files)} {key} dark images.')
        
        return validated_files





class FlatFieldCorrector(Corrector):
    """
    Helper class for performing flat-field corrections.
    """


    def __init__(
        self,
        out_directory: Path | str | None = None,
        data_directory: Path | str | None = None,
        instrument: Instrument = OPTICAM_MX(),
        rebin_factor: int = 1,
        image_filter: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        bias_corrector: BiasCorrector | None = None,
        dark_corrector: DarkNoiseCorrector | None = None,
        ) -> None:
        """
        Initialise the flat-field corrector.
        
        Parameters
        ----------
        out_directory : Path | str | None, optional
            The path to the output directory, by default `None`. Master flat-field images will be saved here.
        data_directory : Path | str | None, optional
            The path to the data directory, by default `None`. This must point to a directory containing a series of
            flat-field images.
        instrument : Instrument, optional
            The instrument, by default `OPTICAM_MX()`.
        rebin_factor : int, optional
            The factor by which to rebin the data, by default 1. Useful if, for example, calibration images were taken
            using a higher resolution than the science images.
        image_filter : Callable[[NDArray[np.float64]], NDArray[np.float64]] | None, optional
            The kernel/filter to apply to calibration images as they are opened. Paez+2026:
            https://ui.adsabs.harvard.edu/abs/2026RASTI...5ag021P/abstract found that a 3x3 median filter
            (e.g., `scipy.ndimage.median_filter()`) can be used to correct for warm pixels in long exposures (> 10 s)
            with OPTICAM.
        bias_corrector : BiasCorrector | None, optional
            The bias corrector to use to bias-correct the flat-field images, by default `None`. If `None`, no bias
            corrections are performed.
        dark_corrector : DarkNoiseCorrector | None, optional
            The dark noise corrector to use to perform dark noise corrections, by default `None`. If `None`, no dark
            noise corrections are performed.
        """
        
        self.bias_corrector = bias_corrector
        self.dark_corrector = dark_corrector
        
        super().__init__(
                         out_directory=out_directory,
                         data_directory=data_directory,
                         instrument=instrument,
                         rebin_factor=rebin_factor,
                         image_filter=image_filter,
                         )


    @property
    def master_image_path(self) -> Path | None:
        """
        The path to the master flat.
        
        Returns
        -------
        Path
            The path to the master flat.
        """
        
        return self.out_directory.joinpath('master_flats.fits.gz') if self.out_directory is not None else None


    def correct(
        self,
        image: NDArray[np.float64],
        camera: str | None = None,
        fltr: str | None = None,
        key: str | None = None,
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Correct an image for flat-fielding.
        
        Parameters
        ----------
        image : NDArray[np.float64]
            The image.
        camera : str
            The camera that took the image.
        fltr : str
            The image filter.
        
        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            The corrected image and the variance of the master flat-field image scaled by the square of the calibrated
            image.
        """
        
        if key is None:
            key = camera_and_filter_key(camera, fltr)
        
        if key not in self.data_files.keys():
            raise ValueError(f"[PHOPTIC] Cannot apply flat-field corrections. No flat-field images found for {key}.")
        
        if key not in self.master_images.keys():
            print(f'[PHOPTIC] {key} master flat-field image not found. Attempting to create.')
            self.create_master_images()
        
        calibrated_image = image / self.master_images[key]
        
        # propagate multiplicative variance
        var = self.master_variances[key] * calibrated_image**2 / self.master_images[key]**2
        
        # correct image for flat-fielding
        return calibrated_image, var


    def create_master_images(
        self,
        overwrite: bool = False,
        ) -> None:
        """
        Create master flat-field images for each filter.
        
        Parameters
        ----------
        bias_corrector : BiasCorrector | None, optional
            The bias corrector.
        overwrite : bool, optional
            Whether to overwrite the existing master flat-field image, by default `False`.
        """
        
        if self.master_image_path.is_file() and not overwrite:
            print(f'[PHOPTIC] Master flats file already exists. To overwrite existing flats, set overwrite=True.')
            return
        
        if not self.passed_checks:
            if self.dark_corrector is not None:
                valid, flat_exptime, dark_exptime = self._dark_corrector_is_valid()
                if not valid:
                    raise ValueError(f'[PHOPTIC] inconsistent exposure times between flat-field images and dark images. Flat-field exposure time: {flat_exptime} s; dark exposure time: {dark_exptime} s.')
        
        for key in self.data_files.keys():
            
            if len(self.data_files[key]) == 1:
                raise Exception(f"[PHOPTIC] Only one {key} flat found. Master flats cannot be created from a single image.")
            
            # read flats
            flats = []
            for flat_file in self.data_files[key]:
                flat, header = flat_file.get_data_and_header()
                
                if self.image_filter is not None:
                    flat = self.image_filter(flat)
                
                if self.rebin_factor > 1:
                    flat = rebin_image(image=flat, factor=self.rebin_factor)
                
                if self.bias_corrector is not None:
                    flat, bias_var = self.bias_corrector.correct(
                        image=flat,
                        camera=camera_key(key),
                        )
                else:
                    bias_var = 0.
                
                if self.dark_corrector is not None:
                    flat, dark_var = self.dark_corrector.correct(
                        image=flat,
                        key=key,
                        dark_flux=self.instrument.get_dark_flux(header=header),
                        )
                else:
                    dark_var = 0.
                
                flats.append(flat)
            
            # use median to account for outliers
            raw_master_flat = np.median(flats, axis=0)
            norm = np.median(raw_master_flat)
            
            self.master_images[key] = raw_master_flat / norm
            self.master_variances[key] = np.pi / (2 * len(flats)) * (np.var(flats, axis=0, ddof=1) + bias_var + dark_var) / norm**2
        
        print('[PHOPTIC] Master flat-field image(s) created.')
        
        self._save_master_image(overwrite=overwrite)
        
        print(f'[PHOPTIC] Master flat-field image(s) saved to {self.master_image_path}.')


    def run_checks(
        self,
        data_files_by_key: dict[str, list[MEFSlice]],
        return_errors: bool = False,
        ) -> None | int:
        """
        Run a series of checks on the corrector to ensure that it is compatible with the data. In this case, check the
        binning of the science images can be matched to those of the flats (accounting for self.rebin_factor), and that
        the there are no missing filters.
        
        Parameters
        ----------
        data_files_by_key : dict[str, list[MEFSlice]]
            The science images grouped by camera:filter keys.
        return_errors : bool, optional
            Whether to return the number of errors raised, by default `False`.
        
        Returns
        -------
        None | int
            If `return_errors=True`, the number of errors raised is returned. Otherwise, nothing is returned.
        """
        
        flat_file = next(iter(self.data_files.values()))[0]  # get the path to a random flat
        flat_header = flat_file.get_header()
        image_file = next(iter(data_files_by_key.values()))[0]  # get the path to a science image
        image_header = image_file.get_header()
        
        science_keys = sorted(list(data_files_by_key.keys()))
        flat_keys = sorted(list(self.data_files.keys()))
        
        errors = 0
        warnings = 0
        
        #################################################### errors ####################################################
        
        # check all science image keys have corresponding flats
        if not all([science_key in flat_keys for science_key in science_keys]):
            errors += 1
            print(f'[PHOPTIC] ERROR: inconsistent keys found between the flat-field images and the science images. Flat-field image keys: ({','.join(flat_keys)}); science image keys: ({','.join(science_keys)})')
        
        # if dark noise corrector defined, check dark images have same exposure times as flats
        if self.dark_corrector is not None:
            valid, flat_exptime, dark_exptime = self._dark_corrector_is_valid()
            if not valid:
                errors += 1
                print(f'[PHOPTIC] ERROR: inconsistent exposure times between flat-field images and dark images. Flat-field exposure time: {flat_exptime} s; dark exposure time: {dark_exptime} s.')
        
        ################################################### warnings ###################################################
        
        # check binnings match
        flat_binning = self.instrument.get_binning(header=flat_header)
        science_binning = self.instrument.get_binning(header=image_header)
        if flat_binning != science_binning:
            warnings += 1
            print(f'[PHOPTIC] WARNING: inconsistent binning found between the flat-field images and the science images. Flat-field image binning: {flat_binning}; science image binning: {science_binning}. If you have passed a suitable rebin_factor to your FlatFieldCorrector instance, then you can safely ignore this warning.')
        
        ################################################### summary ###################################################
        
        if errors > 0 or warnings > 0:
            print()  # blank line for readibility
        
        if errors == 0:
            print(f'[PHOPTIC] FlatFieldCorrector sucessfully passed all checks.')
        else:
            if errors == 1:
                print('[PHOPTIC] FlatFieldCorrector failed 1 check.')
            else:
                print(f'[PHOPTIC] FlatFieldCorrector failed {errors} checks.')
        
        if warnings == 1:
            print('[PHOPTIC] FlatFieldCorrector triggered a warning during 1 check. Warnings may be ignored provided their caveats are satisfied.')
        elif warnings > 1:
            print(f'[PHOPTIC] FlatFieldCorrector triggered a warning during {warnings} checks. Warnings may be ignored provided their caveats are satisfied.')
        
        if errors == 0:
            self.passed_checks = True
        
        if return_errors:
            return errors


    def _validate_data(
        self,
        files: list[MEFSlice],
        ) -> dict[str, list[MEFSlice]]:
        """
        Ensure that the flat-field images in the specified directory are valid (i.e., all use the same binning).
        
        Parameters
        ----------
        file_paths : list[MEFSlice]
            The flat-field image files.
        
        Returns
        -------
        dict[str, list[MEFSlice]]
            A dictionary containing the paths to the flat-field image files grouped by each filter.
        """
        
        keys: dict[str, str] = {}
        binnings: dict[str, str] = {}
        exptimes: dict[str, float] = {}
        validated_files: dict[str, list[MEFSlice]] = {}
        
        for file in files:
            header = file.get_header()
            
            fltr = self.instrument.get_filter(header=header)
            camera = self.instrument.get_camera(header=header)
            key = camera_and_filter_key(camera, fltr)
            
            keys[file.key] = key
            binnings[file.key] = self.instrument.get_binning(header=header)
            exptimes[file.key] = float(header[self.instrument.exptime_kw])
            
            if key not in validated_files.keys():
                validated_files[key] = []
            validated_files[key].append(file)
        
        if len(set(binnings.values())) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='binnings.json',
                file_contents=binnings,
                )
            raise ValueError(f'[PHOPTIC] Inconsistent binning detected in the flat-field images. Image binnings have been logged to {self.out_directory.joinpath('diag/binnings.json')}')
        
        if len(set(exptimes.values())) > 1:
            log_file(
                out_directory=self.out_directory,
                file_name='exptimes.json',
                file_contents=exptimes,
                )
            raise ValueError(f'[PHOPTIC] Inconsistent exposure times detected in the flat-field images. Exposure times have been logged to {self.out_directory.joinpath('diag/exptimes.json')}')
        
        for key, valid_files in validated_files.items():
            print(f'[PHOPTIC] {len(valid_files)} {key} flat-field images.')
        
        return validated_files


    def _dark_corrector_is_valid(self) -> tuple[bool, float, float]:
        """
        Check that the dark images have the same exposure time as the flat-field images.
        
        Returns
        -------
        tuple[bool, float, float]
            If the exposure times are equal, returns `True, 0., 0.,`. Otherwise, returns `False, flat_exposure_time, 
            dark_exposure_time`.
        """
        
        if self.dark_corrector.data_files is not None:
            dark_file = next(iter(self.dark_corrector.data_files.values()))[0]
            dark_header = dark_file.get_header()
            dark_exptime = float(dark_header[self.instrument.exptime_kw])
            
            flat_file = next(iter(self.data_files.values()))[0]
            flat_header = flat_file.get_header()
            flat_exptime = float(flat_header[self.instrument.exptime_kw])
            
            if dark_exptime != flat_exptime:
                return False, flat_exptime, dark_exptime
        
        return True, 0., 0.




