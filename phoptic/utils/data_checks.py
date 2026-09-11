from functools import partial
from logging import Logger
from multiprocessing import cpu_count
from pathlib import Path
import re
import warnings


import numpy as np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


from phoptic.utils.constants import bar_format
from phoptic.utils.fits_handlers import get_header_info
from phoptic.mef_slice import create_file_paths, MEFSlice
from phoptic.utils.helpers import camera_and_filter_key, sort_dict_by_filters
from phoptic.utils.logging import log_file
from phoptic.instruments import Instrument




def scan_data(
    out_directory: Path | str,
    data_directory: Path | str,
    instrument: Instrument,
    barycenter: bool = True,
    verbose: bool = True,
    return_output: bool = False,
    logger: Logger | None = None,
    number_of_processors = cpu_count() // 2,
    ) -> None | tuple[dict[str, list[MEFSlice]], int, dict[str, float], list[MEFSlice], float]:
    """
    Check that the data are self-consistent.
    
    Parameters
    ----------
    out_directory : Path | str
        The path to the directory in which output files will be saved.
    data_directory : Path | str
        The path to the directory containing the data.
    instrument : Instrument
        The instrument that produced the data.
    barycenter : bool, optional
        Whether to apply a Barycentric correction to the image time stamps, by default `True`. Only relevant if
        `return_output=True`.
    verbose : bool, optional
        Whether to print any output info, by default `True`.
    return_output : bool, optional
        Whether to return any output, by default `False`.
    logger : Logger | None, optional
        The logger, by default `None`.
    number_of_processors : _type_, optional
        The number of processors to use, by default `cpu_count() // 2`.
    
    Returns
    -------
    None | tuple[dict[str, list[MEFSlice]], int, dict[str, float], list[MEFSlice], float]:
        If `return_output=True`, the files grouped by camera, binning scale, Barycentric MJD dates, ignored files, and
        the reference date are returned. Otherwise, nothing is returned.
    """
    
    out_directory = Path(out_directory)
    data_directory = Path(data_directory)
    
    files: list[MEFSlice] = create_file_paths(data_directory)
    
    # check instrument
    errors = instrument.run_checks(
        files[0],
        return_errors=True,
        )
    if errors == 1:
        raise ValueError(f'[PHOPTIC] {errors} Instrument error needs to be resolved.')
    elif errors > 1:
        raise ValueError(f'[PHOPTIC] {errors} Instrument errors need to be resolved.')
    
    camera_files: dict[str, list[MEFSlice]] = {}  # {filter : [files]}
    
    # scan files
    chunksize = max(1, len(files) // 100)  # set chunksize to ~1% of the number of files
    results = process_map(
        partial(
            get_header_info,
            instrument=instrument,
            barycenter=barycenter,
            ),
        files,
        max_workers=number_of_processors,
        disable=not verbose,
        desc="[PHOPTIC] Parsing file headers",
        chunksize=chunksize,
        bar_format=bar_format,
        tqdm_class=tqdm)
    
    # unpack results
    binning, bmjds, cameras, filters, ignored_files = parse_header_results(
        results=results,
        files=files,
        out_directory=out_directory,
        logger=logger,
        )
    
    # for each unique filter
    for fltr in set(filters.values()):
        for camera in set(cameras.values()):
            key = camera_and_filter_key(camera, fltr)
            camera_files.update({key: []})  # prepare dictionary entry
    
    for file in files:
        if file not in ignored_files:
            key = camera_and_filter_key(cameras[file.key], filters[file.key])
            camera_files[key].append(file)  # add file name to dict list
    
    # remove keys with no files
    for key in list(camera_files.keys()):
        if len(camera_files[key]) == 0:
            camera_files.pop(key)
    
    camera_files: dict[str, list[MEFSlice]] = sort_dict_by_filters(camera_files)
    
    # sort files by time
    for key in list(camera_files.keys()):
        camera_files[key].sort(key=lambda x: bmjds[x.key])  # use MEFSlice's key attribute to avoid unhashable error
    
    t_ref = min(list(bmjds.values()))  # get reference BMJD
    
    if logger is not None:
        logger.info(f'Binning: {binning}')
        logger.info(f'Filters: {", ".join(list(camera_files.keys()))}')
        for fltr in list(camera_files.keys()):
            logger.info(f'{len(camera_files[fltr])} {fltr} images.')
    elif verbose:
        print(f'Binning: {binning}')
        print(f'Filters: {", ".join(list(camera_files.keys()))}')
        for fltr in list(camera_files.keys()):
            print(f'{len(camera_files[fltr])} {fltr} images.')
    
    if return_output:
        return camera_files, get_binning_scale(binning), bmjds, ignored_files, t_ref


def parse_header_results(
    results: tuple[list[float], list[float], list[str], list[str], list[float]],
    files: list[MEFSlice],
    out_directory: Path,
    logger: Logger | None,
    ) -> tuple[str, dict[str, float], dict[str, str], dict[str, str], list[MEFSlice]]:
    """
    Parse the header info results.
    
    Parameters
    ----------
    results : tuple[list[float], list[float], list[str], list[str], list[float]]
        The header info results.
    files : list[MEFSlice]
        The list of `MEFSlice` instances representing each image.
    out_directory : str
        The directory path to which any output files will be saved.
    logger : Logger | None
        The logger.
    
    Returns
    -------
    tuple[str, dict[str, float], dict[str, str], dict[str, str], list[MEFSlice]]
        The binning scale, BMJD dates, cameras, filters, and ignored files.
    
    Raises
    ------
    ValueError
        If more than three filters are detected.
    ValueError
        If more than one binning mode is detected.
    """
    
    binnings: dict[str, str] = {}
    bmjds: dict[str, float] = {}
    exposures: dict[str, float] = {}
    cameras: dict[str, str] = {}
    filters: dict[str, str] = {}
    ignored_files: list[MEFSlice] = []
    
    # unpack results
    raw_bmjds, raw_exposures, raw_cameras, raw_filters, raw_binnings = zip(*results)
    
    # consolidate results
    for i in range(len(raw_bmjds)):
        if raw_bmjds[i] is not None:
            key = files[i].key
            binnings.update({key: raw_binnings[i]})
            bmjds.update({key: raw_bmjds[i]})
            exposures.update({key: raw_exposures[i]})
            filters.update({key: raw_filters[i]})
            cameras.update({key: raw_cameras[i]})
        else:
            ignored_files.append(files[i])
    
    # get unique filters
    unique_filters = set(filters.values())
    
    # ensure there is at most one type of binning
    unique_binnings = set(binnings.values())
    if len(unique_binnings) > 1:
        log_file(
            out_directory=out_directory,
            file_name='binnings.json',
            file_contents=binnings,
            )
        string = f"Inconsistent binning detected. All images must have the same binning. Image binnings have been logged to {out_directory.joinpath('diag/binnings.json')}."
        if logger is not None:
            logger.error(string)
        raise ValueError(string)
    elif len(unique_binnings) == 0:
        raise ValueError(f'[PHOPTIC] No binning values detected.')
    unique_binning = unique_binnings.pop()  # get unique binning
    
    # check for large differences in time
    for fltr in unique_filters:
        fltr_bmjds = np.sort(np.array([bmjds[file.key] for file in files if file.key in filters and filters[file.key] == fltr]))
        t = fltr_bmjds - np.min(fltr_bmjds)
        dt = np.diff(t) * 86400
        if np.any(dt > 10 * np.median(dt)):
            indices = np.where(dt > 10 * np.median(dt))[0]
            for index in indices:
                string = f"Large time gap detected between {files[index].path.name} and {files[index + 1].path.name} ({dt[index]:.3f} s compared to the median time difference of {np.median(dt):.3f} s). This may cause alignment issues. If so, consider moving all files after this gap to a separate directory."
                warnings.warn(string)
                if logger is not None:
                    logger.warning(string)
    
    # check all images use the same exposure time
    if len(set(exposures.values())) > 1:
        log_file(
            out_directory=out_directory,
            file_name='exposures.json',
            file_contents=exposures,
            )
        string = f'Inconsistent exposure times detected. This is not necessarily a problem, but may cause issues with Fourier transforms later on. Image exposure times have been logged to {out_directory.joinpath('diag/exposures.json')}.'
        if logger is not None:
            logger.error(string)
        warnings.warn(string)
    
    return unique_binning, bmjds, cameras, filters, ignored_files


def get_binning_scale(binning: str) -> int:
    """
    Given a binning mode string, extract the x and y binning scales as integers.
    
    Parameters
    ----------
    binning : str
        The binning mode string (e.g., "2x2", "1 2", etc.). The first number is assumed to be the binning scale in x,
        while the second number is assumed to be the binning scale in y.
    
    Returns
    -------
    int
        The binning scale.
    """
    
    x, y = map(int, re.findall(r"\d+", binning))
    
    assert(x == y), f'[PHOPTIC] Anisotropic binning detected: {binning}. Currently, OPTICAM only supports isotropic binning modes. We apologise for the inconvenience.'
    
    return x
















