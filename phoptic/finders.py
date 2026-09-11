from typing import Callable


from astropy.table import QTable
import numpy as np
from numpy.typing import NDArray
from photutils.background import Background2D
from photutils.segmentation import SourceCatalog, SourceFinder


from phoptic.background.global_background import BaseBackground
from phoptic.utils.constants import fwhm_scale




class DefaultFinder:
    """
    Default source finder. Combines image segmentation with source deblending.
    """
    
    def __init__(
        self,
        n_pixels: int,
        border_width: int = 0,
        ):
        """
        Default source finder. Combines image segmentation with source deblending.
        
        Parameters
        ----------
        n_pixels : int
            The minimum number of connected source pixels.
        border_width : int, optional
            Sources within this many pixels of the border will be ignored, by default 0 (no sources are ignored).
        """
        
        assert type(n_pixels) is int and n_pixels > 0, '[PHOPTIC] npixels must be a positive integer.'
        
        self.border_width = border_width
        self.finder = SourceFinder(n_pixels=n_pixels, progress_bar=False)
    
    def __call__(
        self,
        data: NDArray,
        threshold: float | NDArray,
        ) -> QTable:
        
        segment_map = self.finder(data, threshold)
        
        if self.border_width > 0:
            segment_map.remove_border_labels(border_width=self.border_width, relabel=True)
        
        tbl = SourceCatalog(data, segment_map).to_table()
        tbl.sort('segment_flux', reverse=True)
        
        # reset label to label sources in order of flux
        tbl['label'] = range(1, len(tbl) + 1)
        
        return tbl




def get_source_coords_from_image(
    image: NDArray,
    finder: DefaultFinder,
    threshold: float | int,
    bkg: Background2D | None = None,
    n_sources: int | None = None,
    background: BaseBackground | None = None,
    return_fwhm: bool = False,
    aperture_selector: Callable[[NDArray[np.float64]], float] | None = None,
    ) -> NDArray | tuple[NDArray, float]:
    """
    Get an array of source coordinates from an image in descending order of source brightness.
    
    Parameters
    ----------
    image : NDArray
        The **non-background-subtracted** image from which to extract source coordinates.
    finder : DefaultFinder
        The source finder.
    threshold : float | int
        The source detection threshold in units of background RMS.
    bkg : Background2D, optional
        The background of the image, by default `None`. If `None`, the background is estimated from the image.
    n_sources : int, optional
        The number of source coordinates to return, by default `None` (all sources will be returned).
    background: BaseBackground | None, optional
        The background estimator if `bkg = None`, by default `None`. Either `bkg` or `background` must be defined.
    return_fwhm : bool, optional
        Whether to return the average PSF FWHM of the image, by default `False`.
    aperture_selector : Callable[[NDArray[np.float64]], float] | None, optional
        The function to use to compute the average PSF FWHM, by default `None`. If a function is passed, it must take
        an array of floats and return a float (e.g., `np.median`).
    
    Returns
    -------
    NDArray
        The source coordinates in descending order of brightness.
    """
    
    if bkg is None and background is not None:
        bkg = background(image)  # get background
    elif bkg is None and background is None:
        raise ValueError('[PHOPTIC] get_source_coords_from_image() requires either bkg or background be specified.')
    
    image_clean = image - bkg.background
    
    tbl = finder(image_clean, threshold*bkg.background_rms)
    
    coords = np.array([tbl["x_centroid"], tbl["y_centroid"]]).T
    
    if n_sources is not None:
        coords = coords[:n_sources]
    
    if return_fwhm:
        fwhm = fwhm_scale * aperture_selector(tbl['semimajor_axis'].value)
        
        return coords, fwhm
    
    return coords


