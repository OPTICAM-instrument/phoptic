"""
Collection of random helper functions.
"""

from pathlib import Path
import re
from typing import Any, Dict, List


from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import numpy as np
from numpy.typing import NDArray
from matplotlib.figure import Figure


from phoptic.utils.constants import filter_order




def camel_to_snake(
    string: str,
    ) -> str:
    """
    Convert a camelCase string to snake_case.
    
    Parameters
    ----------
    string : str
        The camelCase string to convert.
    
    Returns
    -------
    str
        The converted snake_case string.
    """

    return re.sub(r'(?<!^)(?=[A-Z])', '_', string).lower()


def sort_dict_by_filters(
    d: Dict[str, Any],
    ) -> Dict[str, Any]:
    """
    Attempt to sort a dictionary whose keys are filter names in order of increasing wavelength (e.g., u, g, r, i, z). If
    unrecognised filters are passed, no sorting is performed.
    
    Parameters
    ----------
    d : Dict[str, Any]
        A dictionary with filter names as keys.
    
    Returns
    -------
    Dict[str, Any]
        The sorted dictionary.
    """
    
    for key in d.keys():
        fltr = filter_key(key)
        if fltr not in filter_order.keys():
            # unrecognised filter; cannot sort
            return d
    
    return dict(sorted(d.items(), key=lambda x: filter_order[filter_key(x[0])]))


def sort_filters(
    filters: List[str],
    ) -> List[str]:
    """
    Attempt to sort a list of filters in order of increasing wavelength (e.g., u, g, r, i, z). If unrecognised filters
    are passed, no sorting is performed.
    
    Parameters
    ----------
    filters : List[str]
        The list of filters.
    
    Returns
    -------
    List[str]
        The sorted list of filters.
    """
    
    for fltr in filters:
        if fltr not in filter_order.keys():
            # unrecognised filter; cannot sort
            return filters
    
    return sorted(filters, key=lambda x: filter_order[x.split(':')[-1]])


def delete_keys_from_nested_dict(
    d: dict[str, Any],
    keys: set[str],
    ) -> None:
    """
    Delete keys from a dictionary in-place.
    
    Parameters
    ----------
    d : dict[str, Any]
        The dictionary. May contain nested dictionaries.
    keys : set[str]
        The keys to remove from the dictionary.
    """
    
    d_copy = d.copy()
    
    for key in d_copy.keys():
        if key in keys:
            d.pop(key)
        elif isinstance(d_copy[key], dict):
            delete_keys_from_nested_dict(d[key], keys)


def match_dict_keys(
    d: dict[Any, Any],
    d_ref: dict[Any, Any],
    ) -> dict[Any, Any]:
    """
    Match the keys of `d` to those of `d_ref`.
    
    Parameters
    ----------
    d : dict[Any, Any]
        The dictionary.
    d_ref : dict[Any, Any]
        The reference dictionary.
    
    Returns
    -------
    dict[Any, Any]
        A copy of `d` whose keys match those of `d_ref`.
    """
    
    new_d = d.copy()
    missing_keys = set()
    for key in new_d.keys():
        if key not in d_ref.keys():
            missing_keys.add(key)
    for key in missing_keys:
        new_d.pop(key)
    
    return new_d


def combine_variances(
    data: NDArray,
    bias_var: float | NDArray[np.float64],
    dark_var: float | NDArray[np.float64],
    flat_var: float | NDArray[np.float64],
    background_rms: float | NDArray,
    read_noise: float,
    ) -> NDArray[np.float64]:
    """
    Compute the propagated variance image.
    
    Parameters
    ----------
    data : NDArray
        The calibrated, background-subtracted image.
    bias_var : float | NDArray[np.float64]
        The bias correction variance term.
    dark_var : float | NDArray[np.float64]
        The dark noise correction variance term.
    flat_var : float | NDArray[np.float64]
        The flat-field correction variance term scaled by the square of the calibrated image.
    background_rms : float | NDArray
        The background RMS.
    read_noise : float
        The read noise [electrons/pixel].
    
    Returns
    -------
    NDArray[np.float64]
        The propagated variance image.
    """
    
    total_variance = np.clip(data, 0., None)  # shot noise
    total_variance += background_rms**2
    total_variance += read_noise**2
    total_variance += bias_var
    total_variance += dark_var
    total_variance += flat_var
    
    return total_variance


def camera_and_filter_key(
    camera: str,
    fltr: str,
    ) -> str:
    """
    Create a unique camera:filter key. This unique key breaks degeneracies in multi-camera, multi-filter instruments,
    such that flat-field corrections can be applied properly.
    
    Parameters
    ----------
    camera : str
        The camera. For single-camera instruments, this can simply be the name of the instrument. For multi-camera
        instruments, however, this value should be unambiguous (e.g., the individual camera name or number).
    fltr : str
        The filter.
    
    Returns
    -------
    str
        The unique camera:filter key.
    """
    
    return camera + ':' + fltr


def camera_key(
    key: str,
    ) -> str:
    """
    Given a unique camera:filter key, get the camera. This is used to apply bias and dark noise corrections, which are
    indifferent to the filter used.
    
    Parameters
    ----------
    key : str
        The unique camera:filter key.
    
    Returns
    -------
    str
        The camera.
    """
    
    return key.split(':')[0]


def filter_key(
    key: str,
    ) -> str:
    """
    Given a unique camera:filter key, get the filter.
    
    Parameters
    ----------
    key : str
        The unique camera:filter key.
    
    Returns
    -------
    str
        The filter.
    """
    
    return key.split(':')[-1]


def save_figure(
    fig: Figure,
    path: Path | str,
    ) -> None:
    """
    Save a figure to the specified path.
    
    Parameters
    ----------
    fig : Figure
        The figure.
    path : Path | str
        The path, including the file name and extension.
    """
    
    fig.savefig(
        path,
        bbox_inches='tight',
        )
    print(f'[PHOPTIC] Plot saved to {Path(path).resolve()}.')


def compute_airmass(
    coords: SkyCoord,
    times: Time,
    observatory: EarthLocation,
    ) -> NDArray:
    """
    Estimate the airmass of an observation using the pointing, timestamp, and the location of the observatory.
    
    TODO: add reference.
    
    Parameters
    ----------
    coords : SkyCoord
        The pointing.
    times : Time
        The observatation time(s).
    observatory : EarthLocation
        The location of the observatory.
    
    Returns
    -------
    NDArray
        The estimated airmass.
    """
    
    altaz_frame = AltAz(obstime=times, location=observatory)
    target_altaz = coords.transform_to(altaz_frame)
    
    alt_deg = target_altaz.alt.deg
    airmass = 1 / np.sin(np.radians(alt_deg + 244 / (165 + 47 * alt_deg**1.1)))
    
    return airmass

