import json
import logging
from logging import Logger
import os
from pathlib import Path
import sys
from types import FunctionType
from typing import Any


from phoptic.utils.helpers import camera_key




def configure_logger(
    out_directory: Path,
    verbose: bool,
    ) -> Logger:
    """
    Configure the reduction logger.
    
    Parameters
    ----------
    out_directory : Path
        The path to the directory in which the log file will be written.
    verbose : bool
        Whether to also output to stdout.
    
    Returns
    -------
    Logger
        The logger.
    """
    
    logger = logging.getLogger('PHOPTIC')
    logger.setLevel(logging.DEBUG)
    
    # clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # create file handler
    file_handler = logging.FileHandler(os.path.join(out_directory, 'info.log'))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # create console handler
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(name)s] %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def log_file(
    out_directory: Path,
    file_name: str,
    file_contents: dict[Any, Any],
    ) -> None:
    """
    Log `file_contents` to a JSON file in `out_directory/diag/`.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    file_name : str
        The name of the diagnostic file. A `.json' suffix is automatically added.
    file_contents : dict[Any, Any]
        The contents that will be saved to the file.
    """
    
    dir_path = os.path.join(out_directory, 'diag')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    if not file_name.endswith('.json'):
        file_name += '.json'
    
    file_path = os.path.join(dir_path, file_name)
    
    with open(file_path, "w") as json_file:
        json.dump(
            file_contents,
            json_file,
            indent=4,
            )


def recursive_log(param: Any, depth: int = 0, max_depth: int = 6) -> Any:
    """
    Recursively log parameters.
    
    Parameters
    ----------
    param : Any
        The parameter to log.
    depth : int, optional
        The parameter depth, by default 0.
    max_depth : int, optional
        The maximum parameter depth, by default 6. This prevents infinite recursion.
    
    Returns
    -------
    Any
        The logged parameter.
    """
    
    ignore_keys = {"_hash"}
    
    if depth > max_depth:
        return f"<Max depth ({max_depth}) reached>"
    
    if isinstance(param, FunctionType):
        # return function name
        return param.__name__
    if isinstance(param, (int, float, str, bool, type(None))):
        return param
    if isinstance(param, (list, tuple, set)):
        return type(param)(recursive_log(item, depth + 1, max_depth) for item in param)
    if isinstance(param, dict):
        return {key: recursive_log(value, depth + 1, max_depth) for key, value in param.items() if key not in ignore_keys}
    if hasattr(param, '__dict__'):
        return {key: recursive_log(value, depth + 1, max_depth) for key, value in vars(param).items() if key not in ignore_keys}
    return str(param)


def log_psf_params(
    out_directory: Path,
    psf_params: dict[str, dict[str, float]],
    binning_scale: int,
    rebin_factor: int,
    pixel_scales: dict[str, float],
    ) -> None:
    """
    Log the PSF parameters.
    
    Parameters
    ----------
    out_directory : str
        The path to the output directory.
    psf_params : dict[str, dict[str, float]]
        The PSF parameters {filter: {PSF param: value}}.
    binning_scale : int
        The observation binning scale.
    rebin_factor : int
        The software rebinning factor.
    pixel_scales : dict[str, float]
        The pixel scale for each filter in arcsec/pixel {filter: pixel scale}.
    """
    
    psf_params_full = {}
    
    for key in psf_params.keys():
        # convert from pixels to arcsec
        semimajor_axis_arcsec = psf_params[key]['semimajor_axis'] * binning_scale * rebin_factor * pixel_scales[camera_key(key)]
        semiminor_axis_arcsec = psf_params[key]['semiminor_axis'] * binning_scale * rebin_factor * pixel_scales[camera_key(key)]
        
        psf_params_full[key] = {
            'semimajor_axis_arcsec': semimajor_axis_arcsec,
            'semimajor_axis_pix': psf_params[key]['semimajor_axis'],
            'semiminor_axis_arcsec': semiminor_axis_arcsec,
            'semiminor_axis_pix': psf_params[key]['semiminor_axis'],
        }
    
    # save PSF params to JSON file
    with open(os.path.join(out_directory, f'misc/psf_params.json'), 'w') as file:
        json.dump(psf_params_full, file, indent=4)























