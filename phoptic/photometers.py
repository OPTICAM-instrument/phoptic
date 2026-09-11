from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple


from numba import njit
import numpy as np
from numpy.typing import NDArray
from photutils.aperture import aperture_photometry, EllipticalAperture


from phoptic.background.local_background import BaseLocalBackground
from phoptic.utils.constants import fwhm_scale
from phoptic.utils.helpers import camel_to_snake, combine_variances




class BasePhotometer(ABC):
    """
    Base class for performing photometry on OPTICAM catalogues.
    """


    def __init__(
        self,
        forced: bool = False,
        source_matching_tolerance: float = 3.,
        local_background_estimator: BaseLocalBackground | Callable | None = None,
        ):
        """
        Initialise a photometer.
        
        Parameters
        ----------
        forced : bool, optional
            Whether to performed "forced" photometry, by default `False`. If `True`, the catalog-aligned coordinates
            are used to perform photometry, even in images where the source is not detected, and the resulting light
            curves will be saved with a 'forced' prefix.
        source_matching_tolerance : float, optional
            The tolerance for source position matching in standard deviations (assuming a Gaussian PSF), by default 3.
            This parameter defines how far from the transformed catalogue position a source can be while still being
            considered the same source.
        local_background_estimator : BaseLocalBackground | Callable | None, optional
            The local background estimator to use, by default `None`. If `None`, the catalogue's 2D background estimator
            is used. If not `None`, this will be used instead of the catalogue's 2D background estimator.
        """
        
        self.forced = forced
        self.source_matching_tolerance = source_matching_tolerance
        
        if local_background_estimator is not None:
            assert callable(local_background_estimator), "[PHOPTIC] local_background_estimator must be either None or a callable object."
        
        self.local_background_estimator = local_background_estimator


    @abstractmethod
    def compute(
        self,
        image: NDArray,
        bias_var: float | NDArray,
        dark_var: float | NDArray,
        flat_var: float | NDArray,
        background_rms: NDArray | None,
        cat_coords: NDArray,
        image_coords: NDArray | None,
        psf_params: Dict[str, float],
        read_noise: float,
        rel_scint_noise: float,
        ) -> Dict[str, List]:
        """
        Compute the fluxes of the catalogued sources from the given image.
        
        Parameters
        ----------
        image : NDArray
            The image. If `self.local_background_estimator` is undefined, this image will be background subtracted.
        bias_var : float | NDArray
            The bias correction variance term.
        dark_var : float | NDArray
            The dark noise correction variance term.
        flat_var : float | NDArray
            The flat-field correction variance term.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        cat_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image. If `match_sources` is True, this will be used to match sources in the
            image to sources in the catalogue.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_axis' (in pixels), 'semiminor_axis' (in pixels), and 'orientation' (in
            *degrees*).
        read_noise : float
            The detector's read noise.
        rel_scint_noise : float
            The relative scintillation noise.
        
        Returns
        -------
        Dict[str, List]
            The photometry results.
        """
        
        pass


    def get_position(
        self,
        cat_coords: NDArray,
        image_coords: NDArray | None,
        source_index: int,
        psf_params: Dict[str, float],
        ) -> NDArray | None:
        """
        Get the position of a source in an image.
        
        Parameters
        ----------
        cat_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image.
        source_index : int
            The source index.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_axis' (in pixels), 'semiminor_axis' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        NDArray
            The source coordinates.
        """
        
        if not self.forced:
            return self.get_closest_source(
                cat_coords,
                image_coords,
                source_index,
                psf_params,
                )
        else:
            return cat_coords[source_index]


    def get_closest_source(
        self,
        cat_coords: NDArray,
        image_coords: NDArray | None,
        source_index: int,
        psf_params: Dict[str, float],
        ) -> NDArray | None:
        """
        Given a source, find the closest source in the catalogue.
        
        Parameters
        ----------
        cat_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image.
        source_index : int
            The source index.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_axis' (in pixels), 'semiminor_axis' (in pixels), and 'orientation' (in
            *degrees*).
        
        Returns
        -------
        NDArray | None
            The coordinates of the closest source.
        """
        
        if image_coords is None:
            return None
        
        # get distances between sources and initial position
        distances = np.sqrt((image_coords[:, 0] - cat_coords[source_index][0])**2 + (image_coords[:, 1] - cat_coords[source_index][1])**2)
        
        # if the closest source is further than the specified tolerance
        if np.min(distances) > self.source_matching_tolerance * np.sqrt(psf_params['semimajor_axis']**2 + psf_params['semiminor_axis']**2):
            return None
        else:
            # get the position of the closest source (assumed to be the source of interest)
            return image_coords[np.argmin(distances)]


    def define_results_dict(
        self,
        ) -> Dict[str, List]:
        """
        Define a results dictionary for the photometer depending on whether `local_background_estimator` is defined.
        
        Returns
        -------
        Dict[str, List]
            The results dictionary with keys 'flux', 'flux_err'. If `local_background_estimator` is defined, the
            dictionary will also contain 'bkg' and 'bkg_err'.
        """
        
        results = {
            'flux': [],
            'flux_err': [],
        }
        
        if self.local_background_estimator is not None:
            results['bkg'] = []
            results['bkg_err'] = []
        
        return results


    def pad_results_dict(
        self,
        results: Dict[str, List],
        ) -> Dict[str, List]:
        """
        Pad the results dictionary with None values for flux and flux error, and background and background error if
        `local_background_estimator' is defined. This is used when a source cannot be matched or its position is
        invalid.
        
        Parameters
        ----------
        results : Dict[str, List]
            The results dictionary to pad.
        
        Returns
        -------
        Dict[str, List]
            The padded results dictionary.
        """
        
        results['flux'].append(None)
        results['flux_err'].append(None)
        
        if self.local_background_estimator is not None:
            results['bkg'].append(None)
            results['bkg_err'].append(None)
        
        return results


    def populate_results_dict(
        self,
        results: Dict[str, List],
        phot_function: Callable,
        image: NDArray,
        bias_var: float | NDArray[np.float64],
        dark_var: float | NDArray[np.float64],
        flat_var: float | NDArray[np.float64],
        background_rms: NDArray | None,
        position: NDArray,
        psf_params: Dict[str, float],
        read_noise: float,
        rel_scint_noise: float,
        ) -> Dict[str, List]:
        """
        Populate the results dictionary with the computed flux, flux error, and background (if applicable) using the
        provided photometry function.
        
        Parameters
        ----------
        results : Dict[str, List]
            The results dictionary to populate.
        phot_function : Callable
            The photometry function to use for computing the flux and flux error. This function should take the image,
            image error, position, and PSF parameters as arguments and return the flux and flux error, and optionally
            the background and background error if `local_background_estimator` is defined.
        image : NDArray
            The image.
        bias_var : float | NDArray
            The bias correction variance term.
        dark_var : float | NDArray
            The dark noise correction variance term.
        flat_var : float | NDArray
            The flat-field correction variance term.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        position : NDArray
            The position of the source in the image.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_axis' (in pixels), 'semiminor_axis' (in pixels), and 'orientation' (in
            *degrees*).
        read_noise : float
            The detector's read noise.
        rel_scint_noise : float
            The relative scintillation noise.
        
        Returns
        -------
        Dict[str, List]
            The updated results dictionary with the computed flux, flux error, and background (if applicable).
        """
        
        if self.local_background_estimator is None:
            flux, flux_err = phot_function(
                image=image,
                bias_var=bias_var,
                dark_var=dark_var,
                flat_var=flat_var,
                background_rms=background_rms,
                position=position,
                psf_params=psf_params,
                read_noise=read_noise,
                rel_scint_noise=rel_scint_noise,
            )
            
            results['flux'].append(flux)
            results['flux_err'].append(flux_err)
        else:
            flux, flux_err, background, background_err = phot_function(
                image=image,
                bias_var=bias_var,
                dark_var=dark_var,
                flat_var=flat_var,
                background_rms=background_rms,
                position=position,
                psf_params=psf_params,
                read_noise=read_noise,
                rel_scint_noise=rel_scint_noise,
            )
            
            results['flux'].append(flux)
            results['flux_err'].append(flux_err)
            results['bkg'].append(background)
            results['bkg_err'].append(background_err)
        
        return results


    def get_label(
            self,
            ) -> str:
            """
            Get the label of the photometer for labelling output.
            
            Returns
            -------
            str
                The label.
            """
            
            save_name = camel_to_snake(self.__class__.__name__).replace('_photometer', '')
            
            # change save directory based on photometer settings
            if self.local_background_estimator is not None:
                save_name += '_annulus'
            if self.forced:
                save_name = 'forced_' + save_name
            
            return save_name




class AperturePhotometer(BasePhotometer):
    """
    A photometer for performing aperture photometry.
    """


    def __init__(
        self,
        semimajor_axis: int | None = None,
        semiminor_axis: int | None = None,
        orientation: float | None = None,
        forced: bool = False,
        source_matching_tolerance: float = 3.,
        local_background_estimator: None | BaseLocalBackground = None,
        ):
        """
        Initialise a photometer.
        
        Parameters
        ----------
        semimajor_axis : int | None, optional
            The semi-major axis of the aperture, by default None (set to the FWHM of the PSF).
        semiminor_axis : int | None, optional
            The semi-minor axis of the aperture, by default None (set to the FWHM of the PSF).
        orientation : float, optional
            The orientation of the ellipse, by default None (set based on the averaged PSF orientation).
        forced : bool, optional
            Whether to performed "forced" photometry, by default `False`. If `True`, the catalog-aligned coordinates
            are used to perform photometry, even in images where the source is not detected, and the resulting light
            curves will be saved with a 'forced' prefix.
        source_matching_tolerance : float, optional
            The tolerance for source position matching in standard deviations (assuming a Gaussian PSF), by default 3.
            This parameter defines how far from the transformed catalogue position a source can be while still being
            considered the same source.
        local_background_estimator : None | BaseLocalBackground, optional
            The local background estimator to use, by default `None`. If `None`, the catalogue's 2D background estimator
            is used. If not `None`, this will be used instead of the catalogue's 2D background estimator.
        """
        
        self.semimajor_axis = semimajor_axis
        self.semiminor_axis = semiminor_axis
        self.orientation = orientation
        
        super().__init__(
            forced=forced,
            source_matching_tolerance=source_matching_tolerance,
            local_background_estimator=local_background_estimator,
            )


    def compute(
        self,
        image: NDArray,
        bias_var: float | NDArray,
        dark_var: float | NDArray,
        flat_var: float | NDArray,
        background_rms: NDArray | None,
        cat_coords: NDArray,
        image_coords: NDArray | None,
        psf_params: Dict[str, float],
        read_noise: float,
        rel_scint_noise: float,
        ) -> Dict[str, List]:
        """
        Compute the fluxes of the catalogued sources from the given image.
        
        Parameters
        ----------
        image : NDArray
            The image. If `self.local_background_estimator` is undefined, this image will be background subtracted.
        bias_var : float | NDArray
            The bias correction variance term.
        dark_var : float | NDArray
            The dark noise correction variance term.
        flat_var : float | NDArray
            The flat-field correction variance term scaled by the square of the calibrated image.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        cat_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image. If `match_sources` is True, this will be used to match sources in the
            image to sources in the catalogue.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_axis' (in pixels), 'semiminor_axis' (in pixels), and 'orientation' (in
            *degrees*).
        read_noise : float
            The detector's read noise.
        rel_scint_noise : float
            The relative scintillation noise.
        
        Returns
        -------
        Dict[str, List]
            The photometry results.
        """
        
        results = self.define_results_dict()
        
        for i in range(len(cat_coords)):
            
            # get position of source depending on whether source matching is enabled or not
            position = self.get_position(
                cat_coords,
                image_coords,
                i,
                psf_params,
                )
            
            # if position is None, pad the results dictionary and continue to the next source
            if position is None:
                results = self.pad_results_dict(results)
                continue
            
            # populate the results dictionary with the computed flux, flux error, and background (if applicable)
            results = self.populate_results_dict(
                results=results,
                phot_function=self.compute_aperture_flux,
                image=image,
                bias_var=bias_var,
                dark_var=dark_var,
                flat_var=flat_var,
                background_rms=background_rms,
                position=position,
                psf_params=psf_params,
                read_noise=read_noise,
                rel_scint_noise=rel_scint_noise,
                )
        
        return results


    def compute_aperture_flux(
        self,
        image: NDArray,
        bias_var: float | NDArray,
        dark_var: float | NDArray,
        flat_var: float | NDArray,
        background_rms: NDArray | None,
        position: NDArray,
        psf_params: Dict[str, float],
        read_noise: float,
        rel_scint_noise: float,
        ) -> Tuple[float, float] | Tuple[float, float, float, float]:
        """
        Compute the aperture flux of a source in the image.
        
        Parameters
        ----------
        image : NDArray
            The image.
        bias_var : float | NDArray
            The bias correction variance term.
        dark_var : float | NDArray
            The dark noise correction variance term.
        flat_var : float | NDArray
            The flat-field correction variance term scaled by the square of the calibrated image.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        position : NDArray
            The position of the source.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_axis' (in pixels), 'semiminor_axis' (in pixels), and 'orientation' (in
            *degrees*).
        read_noise : float
            The instrument's read noise.
        rel_scint_noise : float
            The relative scintillation noise.
        
        Returns
        -------
        Tuple[float, float] | Tuple[float, float, float, float, float]
            The flux and its error. If `local_background_estimator` is defined, the local background and its error are
            also returned.
        """
        
        aperture = self.get_aperture(
            position=position,
            psf_params=psf_params,
            )
        
        if self.local_background_estimator is None:
            total_var = combine_variances(
                data=image,
                bias_var=bias_var,
                dark_var=dark_var,
                flat_var=flat_var,
                background_rms=np.asarray(background_rms),
                read_noise=read_noise,
                )
            
            phot_table = aperture_photometry(image, aperture, error=np.sqrt(total_var))
            
            flux = phot_table["aperture_sum"].value[0]
            flux_err = np.sqrt(phot_table["aperture_sum_err"].value[0]**2 + (rel_scint_noise * flux)**2)
            
            return flux, flux_err
        else:
            
            local_background, local_background_rms = self.local_background_estimator(
                image,
                position,
                psf_params['semimajor_axis'],
                psf_params['semiminor_axis'],
                psf_params['orientation'],
                )
            
            data_clean = image - local_background
            total_var = combine_variances(
                data=data_clean,
                bias_var=bias_var,
                dark_var=dark_var,
                flat_var=flat_var,
                background_rms=local_background_rms,
                read_noise=read_noise,
            )
            
            phot_table = aperture_photometry(data_clean, aperture, error=np.sqrt(total_var))
            
            flux = phot_table["aperture_sum"].value[0]
            flux_err = np.sqrt(phot_table["aperture_sum_err"].value[0]**2 + (rel_scint_noise * flux)**2)
            
            return flux, flux_err, float(local_background), float(local_background_rms)


    def get_aperture(
        self,
        position: NDArray,
        psf_params: Dict[str, float],
        ) -> EllipticalAperture:
        
        if self.semimajor_axis is not None and self.semiminor_axis is not None and self.orientation is not None:
            return EllipticalAperture(
                positions=position,
                a=self.semimajor_axis,
                b=self.semiminor_axis,
                theta=self.orientation,
                )
        else:
            return EllipticalAperture(
                positions=position,
                a=fwhm_scale * psf_params['semimajor_axis'],
                b=fwhm_scale * psf_params['semiminor_axis'],
                theta=psf_params['orientation'],
                )


    def get_aperture_area(
        self,
        psf_params: Dict[str, float],
        ) -> float:
        """
        Get the area of the aperture.
        
        Parameters
        ----------
        psf_params : Dict[str, float],
            The PSF parameters.
        
        Returns
        -------
        float
            The area of the aperture.
        """
        
        return self.get_aperture(
            position=np.zeros(2),  # position does not matter
            psf_params=psf_params,
            ).area




class OptimalPhotometer(BasePhotometer):
    """
    A photometer that implements the optimal photometry method described in Naylor 1998, MNRAS, 296, 339-346.
    """


    def compute(
        self,
        image: NDArray,
        bias_var: float | NDArray[np.float64],
        dark_var: float | NDArray[np.float64],
        flat_var: float | NDArray[np.float64],
        background_rms: NDArray | None,
        cat_coords: NDArray,
        image_coords: NDArray | None,
        psf_params: Dict[str, float],
        read_noise: float,
        rel_scint_noise: float,
        ) -> Dict[str, List]:
        """
        Compute the fluxes of the catalogued sources from the given image.
        
        Parameters
        ----------
        image : NDArray
            The image. If `self.local_background_estimator` is undefined, this image will be background subtracted.
        bias_var : float | NDArray
            The bias correction variance term.
        dark_var : float | NDArray
            The dark noise correction variance term.
        flat_var : float | NDArray
            The flat-field correction variance term scaled by the square of the calibrated image.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        cat_coords : NDArray
            The source coordinates in the catalogue.
        image_coords : NDArray | None
            The source coordinates in the image. If `match_sources` is True, this will be used to match sources in the
            image to sources in the catalogue.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_axis' (in pixels), 'semiminor_axis' (in pixels), and 'orientation' (in
            *degrees*).
        read_noise : float
            The detector's read noise.
        rel_scint_noise : float
            The relative scintillation noise.
        
        Returns
        -------
        Dict[str, List]
            The photometry results.
        """
        
        results = self.define_results_dict()
        
        for i in range(len(cat_coords)):
            
            position = self.get_position(
                cat_coords,
                image_coords,
                i,
                psf_params,
            )
            
            if position is None:
                results = self.pad_results_dict(results)
                continue
            
            results = self.populate_results_dict(
                results=results,
                phot_function=self.compute_optimal_flux,
                image=image,
                bias_var=bias_var,
                dark_var=dark_var,
                flat_var=flat_var,
                background_rms=background_rms,
                position=position,
                psf_params=psf_params,
                read_noise=read_noise,
                rel_scint_noise=rel_scint_noise,
                )
        
        return results


    def compute_optimal_flux(
        self,
        image: NDArray,
        bias_var: float | NDArray[np.float64],
        dark_var: float | NDArray[np.float64],
        flat_var: float | NDArray[np.float64],
        background_rms: NDArray | None,
        position: NDArray,
        psf_params: Dict[str, float],
        read_noise: float,
        rel_scint_noise: float,
        ) -> Tuple[float, float] | Tuple[float, float, float, float]:
        """
        Compute the optimal flux of a source in the image as described in Naylor 1998, MNRAS, 296, 339-346.
        
        Parameters
        ----------
        image : NDArray
            The image.
        bias_var : float | NDArray
            The bias correction variance term.
        dark_var : float | NDArray
            The dark noise correction variance term.
        flat_var : float | NDArray
            The flat-field correction variance term scaled by the square of the calibrated image.
        background_rms : NDArray | None
            The background RMS. May be `None` if `self.local_background_estimator` is defined.
        position : NDArray
            The position of the source in the image, given as (y, x) coordinates.
        psf_params : Dict[str, float]
            The PSF parameters for the camera used to take the image. This parameter is defined in the catalogue and
            has the following keys: 'semimajor_axis' (in pixels), 'semiminor_axis' (in pixels), and 'orientation' (in
            *degrees*).
        read_noise : float
            The instrument's read noise.
        rel_scint_noise : float
            The relative scintillation noise.
        
        Returns
        -------
        Tuple[float, float] | Tuple[float, float, float, float]
            The flux and flux error. If `local_background_estimator` is defined, the background and its error are also
            returned.
        """
        
        if self.local_background_estimator is None:
            return get_optimal_flux_and_error(
                image=image,
                bias_var=bias_var,
                dark_var=dark_var,
                flat_var=flat_var,
                background_rms=np.asarray(background_rms),
                read_noise=read_noise,
                position=position,
                psf_params=psf_params,
                rel_scint_noise=rel_scint_noise,
                )
        else:
            # estimate local background using annulus
            local_background, local_background_rms = self.local_background_estimator(
                image,
                position,
                psf_params['semimajor_axis'],
                psf_params['semiminor_axis'],
                psf_params['orientation'],
                )
            
            flux, flux_error = get_optimal_flux_and_error(
                image=image - local_background,
                bias_var=bias_var,
                dark_var=dark_var,
                flat_var=flat_var,
                background_rms=local_background_rms,
                read_noise=read_noise,
                position=position,
                psf_params=psf_params,
                rel_scint_noise=rel_scint_noise,
                )
            
            return flux, flux_error, float(local_background), float(local_background_rms)




@njit
def get_optimal_weights(
    var: NDArray[np.float64],
    position: NDArray[np.float64],
    psf_major: float,
    psf_minor: float,
    psf_orientation: float,
    ) -> Tuple[NDArray[np.float64], float]:
    """
    Compute the optimal weight for each pixel in an image.
    
    Parameters
    ----------
    var : NDArray[np.float64]
        The variance image.
    position : NDArray[np.float64]
        The position of the source.
    psf_major : float
        The semi-major axis of the PSF.
    psf_minor : float
        The semi-minor axis of the PSF.
    psf_orientation : float
        The orientation of the PSF in degrees.
    
    Returns
    -------
    Tuple[NDArray[np.float64], float]
        The weights and the normalisation constant.
    """
    
    # define pixel coordinates
    h, w = var.shape
    y = np.arange(h).reshape((h, 1))
    x = np.arange(w).reshape((1, w))
    
    theta = psf_orientation * np.pi / 180
    
    # offset coordinates to the position of the source and align axes with the orientation of the PSF
    x0, y0 = position
    x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    
    psf = np.exp(- .5 * ((x_rot / psf_major)**2 + (y_rot / psf_minor)**2))
    weights = psf / var
    normalisation = np.sum(psf**2 / var)
    
    return weights, normalisation


def get_optimal_flux_and_error(
    image: NDArray[np.float64],
    bias_var: float | NDArray[np.float64],
    dark_var: float | NDArray[np.float64],
    flat_var: float | NDArray[np.float64],
    background_rms: float | NDArray[np.float64],
    read_noise: float,
    position: NDArray[np.float64],
    psf_params: Dict[str, float],
    rel_scint_noise: float,
    ) -> Tuple[float, float]:
    """
    Compute the optimal flux and its error.
    
    Parameters
    ----------
    image : NDArray[np.float64]
        The background-subtracted image.
    bias_var : float | NDArray
        The bias correction variance term.
    dark_var : float | NDArray
        The dark noise correction variance term.
    flat_var : float | NDArray
        The flat-field correction variance term scaled by the square of the calibrated image.
    background_rms : float | NDArray[np.float64]
        The background RMS. May be a scalar value or an `NDArray` with the same shape as `image`.
    read_noise : float
        The instrument's read noise.
    position : NDArray[np.float64]
        The source position [x, y].
    psf_params : Dict[str, float]
        The PSF parameters.
    rel_scint_noise : float
        The relative scintillation noise.
    
    Returns
    -------
    Tuple[float, float]
        The flux and its corresponding error.
    """
    
    total_var = combine_variances(
        data=image,
        bias_var=bias_var,
        dark_var=dark_var,
        flat_var=flat_var,
        background_rms=background_rms,
        read_noise=read_noise,
        )
    
    weights, norm = get_optimal_weights(
        var=total_var,
        position=position,
        psf_major=psf_params['semimajor_axis'],
        psf_minor=psf_params['semiminor_axis'],
        psf_orientation=psf_params['orientation'],
        )
    
    flux = np.sum(image * weights) / norm
    flux_error = np.sqrt(1 / norm + (rel_scint_noise * flux)**2)
    
    return flux, flux_error


def get_growth_curve(
    image: NDArray,
    x_centroid: float,
    y_centroid: float,
    r_max: int,
    ) -> Tuple[NDArray, NDArray]:
    """
    Compute the growth curve for a point in an image.
    
    Parameters
    ----------
    image : NDArray
        The image.
    x_centroid : float
        The x centroid of the point.
    y_centroid : float
        The y centroid of the point.
    r_max : int
        The maximum radius in pixels.
    
    Returns
    -------
    Tuple[NDArray, NDArray]
        _description_
    """
    
    position = np.array([x_centroid, y_centroid])
    
    radii, fluxes = [], []
    
    for r in range(1, r_max):
        radii.append(r)
        
        photometer = AperturePhotometer(
            semimajor_axis=r,
            semiminor_axis=r,
            orientation=0,
            forced=True,
            )
        
        flux = photometer.compute_aperture_flux(
            image=image,
            bias_var=0.,
            dark_var=0.,
            flat_var=0.,
            background_rms=0.,
            position=position,
            psf_params={},  # empty dict since not needed
            read_noise=0.,
            rel_scint_noise=0.
            )[0]
        
        fluxes.append(flux)
    
    return np.array(radii), np.array(fluxes)