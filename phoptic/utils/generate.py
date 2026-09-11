from pathlib import Path
from typing import Dict, List, Tuple


from astropy.io import fits
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


from phoptic.utils.constants import bar_format




FILTERS: List[str] = ["g", "r", "i"]
N_SOURCES: int = 6

RMS: float = 0.02  # fractional RMS of variability
FREQ: float = 0.135  # frequency of variability

# variability phase lags
PHASE_LAGS: Dict[str, float] = {
    'g': 0,
    'r': np.pi / 2,
    'i': np.pi,
}

MEDIAN_BKG: float = 100.0
MEDIAN_BKG_RMS: float = np.sqrt(MEDIAN_BKG)
MEDIAN_FLAT_FLUX: float = 10000.0
MEDIAN_FLAT_FLUX_RMS: float = np.sqrt(MEDIAN_FLAT_FLUX)




def two_dimensional_gaussian(
    image: NDArray,
    x_centroid: float,
    y_centroid: float,
    flux: float,
    a: float,
    b: float,
    theta: float,
    ) -> NDArray:
    """
    Add a source to an image.
    
    Parameters
    ----------
    image : NDArray
        The image.
    x_centroid : float
        The x-coordinate of the source.
    y_centroid : float
        The y-coordinate of the source.
    flux : float
        The total flux of the source.
    a : float
        The semi-major standard deviation.
    b : float
        The semi-minor standard deviation.
    theta : float
        The rotation angle of the source.
    
    Returns
    -------
    NDArray
        The image with the source added.
    """
    
    theta_rad = theta * np.pi / 180
    
    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    
    dx = x - x_centroid
    dy = y - y_centroid
    
    x_rot = dx * np.cos(theta_rad) + dy * np.sin(theta_rad)
    y_rot = -dx * np.sin(theta_rad) + dy * np.cos(theta_rad)
    
    amp = flux / (2 * np.pi * a * b)
    
    return amp * np.exp(-.5 * (np.square(x_rot / a) + np.square(y_rot / b)))


def variable_function(
    i: float,
    fltr: str,
    ) -> float:
    """
    Variable flux to be added to a source.
    
    Parameters
    ----------
    i : float
        The image index (equivalent to time).
    fltr : str
        The filter of the image, used to introduce a lag between filters.
    
    Returns
    -------
    float
        The flux.
    """
    
    amp = RMS * np.sqrt(2)  # convert RMS amplitude to peak amplitude
    
    return 1 + amp * np.sin(2 * np.pi * i * FREQ + PHASE_LAGS[fltr])


def create_image(
    binning_scale: int,
    ) -> NDArray:
    """
    Create a blank base image.
    
    Parameters
    ----------
    binning_scale : int
        The binning scale of the image.
    
    Returns
    -------
    NDArray
        The blank image.
    """
    
    return np.zeros((2048 // binning_scale, 2048 // binning_scale))


def poisson_noise(
    image: NDArray,
    i: int,
    ) -> NDArray:
    """
    Create a Poisson noise image.
    
    Parameters
    ----------
    image : NDArray
        The science image.
    i : int
        The RNG seed.
    
    Returns
    -------
    NDArray
        The noisy image.
    """
    
    rng = np.random.default_rng(i)
    
    return rng.normal(0, np.sqrt(image))


def noisy_background(
    image: NDArray,
    i: int,
    median: float | NDArray,
    rms: float | NDArray,
    ) -> NDArray:
    """
    Create a noisy background image.
    
    Parameters
    ----------
    image : NDArray
        The science image. Used to determine the output image's shape.
    i : int
        The RNG seed.
    median : float | NDArray
        The median background.
    rms : float | NDArray
        The background RMS.
    
    Returns
    -------
    NDArray
        The noisy background image.
    """
    
    rng = np.random.default_rng(i)
    
    return rng.normal(median, rms, size=image.shape)


def create_images(
    out_directory: Path,
    variable_source: int,
    source_positions: NDArray,
    fluxes: NDArray,
    i: int,
    binning_scale: int,
    circular_aperture: bool,
    overwrite: bool,
    ) -> None:
    """
    Create the ith image for each filter.
    
    Parameters
    ----------
    out_directory : Path
        The directory path to the output.
    variable_source : int
        The index of the variable source.
    source_positions : NDArray
        The positions of the sources.
    fluxes : NDArray
        The fluxes of the sources.
    i : int
        The image index (equivalent to time).
    binning_scale : int
        The binning scale of the image.
    circular_aperture : bool
        Whether to apply a circular aperture shadow to the image.
    overwrite : bool
        Whether to overwrite the image if it already exists.
    """
    
    for fltr in FILTERS:
        save_path = out_directory.joinpath(f'240101{fltr}{200000000 + i}o.fits')
        if save_path.is_file() and not overwrite:
            continue
        
        # generate blank image
        image = create_image(binning_scale)
        
        # PSF parameters (typical PSF stdev of ~6pix at 1x1 binning for decent seeing)
        semimajor_sigma = 6 / (2048 / image.shape[0])
        semiminor_sigma = 6 / (2048 / image.shape[0])
        orientation = 0
        
        # (x, y) translations
        rng = np.random.default_rng(i)
        dx = 2048 // 512 * rng.normal() / binning_scale
        dy = 2048 // 512 * rng.normal() / binning_scale
        x_positions = source_positions[:, 0] + dx
        y_positions = source_positions[:, 1] + dy
        
        # put sources in the image
        for j in range(N_SOURCES):
            if j == variable_source:
                image += two_dimensional_gaussian(
                    image,
                    x_positions[j],
                    y_positions[j],
                    fluxes[j] * variable_function(i, fltr),  # add variable flux to the source
                    semimajor_sigma,
                    semiminor_sigma,
                    orientation,
                    )
            else:
                image += two_dimensional_gaussian(
                    image,
                    x_positions[j],
                    y_positions[j],
                    fluxes[j],
                    semimajor_sigma,
                    semiminor_sigma,
                    orientation,
                    )
        
        image += poisson_noise(image, i)  # add shot noise
        image += noisy_background(image, i, MEDIAN_BKG, MEDIAN_BKG_RMS)  # add background
        
        if circular_aperture:
            image = apply_flat_field(image)  # apply circular aperture shadow
        
        # create fits file
        hdu = fits.PrimaryHDU(image)
        hdu.header["FILTER"] = fltr
        hdu.header["BINNING"] = f'{binning_scale}x{binning_scale}'
        hdu.header["GAIN"] = 1.
        hdu.header["EXPOSURE"] = 1.
        hdu.header["DARKCURR"] = 0.
        hdu.header["INSTRUME"] = 'OPTICAM'
        hdu.header["AIRMASS"] = 1
        
        # create observation pointing
        hdu.header['RA'] = 0.
        hdu.header['DEC'] = 0.
        
        # create observation time
        hh = str(i // 3600).zfill(2)
        mm = str(i % 3600 // 60).zfill(2)
        ss = str(i % 60).zfill(2)
        hdu.header["UT"] = f"2024-01-01 {hh}:{mm}:{ss}"
        
        # save fits file
        hdu.writeto(save_path, overwrite=overwrite)


def apply_flat_field(
    image: NDArray,
    ) -> NDArray:
    """
    Apply a circular aperture shadow to an image.
    
    Parameters
    ----------
    image : NDArray
        The image.
    
    Returns
    -------
    NDArray
        The image with a circular aperture shadow.
    """
    
    # define mask to apply circular aperture
    x_mid, y_mid = image.shape[1] // 2, image.shape[0] // 2
    distance_from_centre = np.sqrt((x_mid - np.arange(image.shape[1]))**2 +
                                   (y_mid - np.arange(image.shape[0]))[:, np.newaxis]**2)
    radius = image.shape[0] // 2
    mask = distance_from_centre >= radius
    
    # create circular aperture shadow
    falloff = 1 / (distance_from_centre[mask] / radius)**2
    
    # apply circular aperture shadow
    image[mask] *= falloff
    
    return image


def create_flats(
    out_directory: Path,
    filters: list,
    i: int,
    binning_scale: int,
    overwrite: bool,
    ) -> None:
    """
    Create the ith flat-field image for each filter. 
    
    Parameters
    ----------
    out_directory : Path
        The directory to save the flat-field images.
    filters : list
        The filters to create flat-field images for.
    i : int
        The index of the flat-field image (equivalent to time).
    binning_scale : int
        The binning scale of the flat-field image.
    overwrite : bool
        Whether to overwrite the flat-field image if it already exists.
    """
    
    for fltr in filters:
        
        save_path = out_directory.joinpath(f'{fltr}-band_image_{i}.fits.gz')
        if save_path.is_file() and not overwrite:
            continue
        
        image = create_image(binning_scale)
        image += noisy_background(
            image=image,
            i=123 * (i + 123),
            median=MEDIAN_FLAT_FLUX,
            rms=MEDIAN_FLAT_FLUX_RMS,
            )
        image = apply_flat_field(image)  # apply circular aperture shadow
        
        # create fits file
        hdu = fits.PrimaryHDU(image)
        hdu.header["FILTER"] = fltr
        hdu.header['EXPOSURE'] = 0.05
        hdu.header["BINNING"] = f'{binning_scale}x{binning_scale}'
        hdu.header["GAIN"] = 1.
        hdu.header["INSTRUME"] = 'OPTICAM'
        hdu.header["AIRMASS"] = 1
        
        # create observation time
        hh = str(i // 3600).zfill(2)
        mm = str(i % 3600 // 60).zfill(2)
        ss = str(i % 60).zfill(2)
        hdu.header["UT"] = f"2024-01-01 {hh}:{mm}:{ss}"
        
        # save fits file
        hdu.writeto(save_path, overwrite=overwrite)


def generate_flats(
    out_directory: Path | str,
    n_flats: int = 5,
    binning_scale: int = 4,
    overwrite: bool = False,
    ) -> None:
    """
    Create synthetic flat-field images.
    
    Parameters
    ----------
    out_directory : Path | str
        The directory to save the data.
    n_flats : int, optional
        The number of flats per camera, by default 5.
    binning_scale : int, optional
        The binning scale of the flat-field images, by default 4 (512x512).
    overwrite : bool, optional
        Whether to overwrite data if they currently exist, by default False.
    """
    
    out_directory = Path(out_directory)
    
    # create directory if it does not exist
    if not out_directory.is_dir():
        out_directory.mkdir(parents=True)
    
    filters = ["g", "r", "i"]
    
    for i in tqdm(range(n_flats), desc="Generating flats", bar_format=bar_format):
        create_flats(
            out_directory,
            filters,
            i,
            binning_scale,
            overwrite,
            )


def setup_obs(
    out_directory: Path,
    binning_scale: int,
    ) -> Tuple[int, NDArray, NDArray]:
    """
    Configure the dummy observation parameters.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    binning_scale : int
        The image binning scale.
    
    Returns
    -------
    Tuple[List[str], int, int, NDArray, NDArray]
        The variable source index, source positions, and peak fluxes.
    """
    
    # create directory if it does not exist
    if not out_directory.is_dir():
        out_directory.mkdir(parents=True)
    
    rng = np.random.default_rng(123)
    
    border = 2048 // (16 * binning_scale)
    source_positions = rng.uniform(border, 2048 // binning_scale - border, (N_SOURCES, 2))  # generate random source positions away from the edges
    
    random_fluxes = np.round(10**(3 + rng.random(N_SOURCES)))  # generate random fluxes (uniform in logarithm)
    fluxes = -np.sort(-random_fluxes)  # sort fluxes in descending order
    
    variable_source = 1  # index of the variable source
    
    print(f'[PHOPTIC] variable source is at ({source_positions[variable_source][0]:.0f}, {source_positions[variable_source][1]:.0f})')
    print(f'[PHOPTIC] variability RMS: {RMS} %')
    print(f'[PHOPTIC] variability frequency: {FREQ} Hz')
    print('[PHOPTIC] variability phase lags:')
    for fltr, lag in PHASE_LAGS.items():
        print(f'    [PHOPTIC] {fltr}-band: {lag:.3f} radians')
    
    # print(f'[PHOPTIC] source fluxes: {fluxes}')
    
    return variable_source, source_positions, fluxes


def generate_observations(
    out_directory: Path | str,
    n_images: int = 100,
    circular_aperture: bool = True,
    binning_scale: int = 4,
    overwrite: bool = False,
    ) -> None:
    """
    Create synthetic observation data for testing and following the tutorials.
    
    Parameters
    ----------
    out_directory : Path | str
        The directory to save the data.
    n_images : int, optional
        The number of images to create, by default 100.
    circular_aperture : bool, optional
        Whether to apply a circular aperture shadow to the images, by default True.
    binning_scale : int, optional
        The binning scale of the images, by default 4 (512x512).
    overwrite : bool, optional
        Whether to overwrite data if they currently exist, by default False.
    """
    
    out_directory = Path(out_directory)
    
    variable_source, source_positions, fluxes = setup_obs(
        out_directory=out_directory,
        binning_scale=binning_scale,
        )
    
    for i in tqdm(range(n_images), desc="Generating observations", bar_format=bar_format):
        create_images(
            out_directory=out_directory,
            variable_source=variable_source,
            source_positions=source_positions,
            fluxes=fluxes,
            i=i,
            binning_scale=binning_scale,
            circular_aperture=circular_aperture,
            overwrite=overwrite,
            )


def generate_gappy_observations(
    out_directory: Path | str,
    n_images: int = 1000,
    circular_aperture: bool = True,
    binning_scale: int = 4,
    overwrite: bool = False,
    ) -> None:
    """
    Create synthetic observation data for testing and following the tutorials.
    
    Parameters
    ----------
    out_directory : Path | str
        The directory to save the data.
    n_images : int, optional
        The number of images to create, by default 100.
    circular_aperture : bool, optional
        Whether to apply a circular aperture shadow to the images, by default True.
    binning_scale : int, optional
        The binning scale of the images, by default 4 (512x512).
    overwrite : bool, optional
        Whether to overwrite data if they currently exist, by default False.
    """
    
    out_directory = Path(out_directory)
    
    variable_source, source_positions, fluxes = setup_obs(
        out_directory=out_directory,
        binning_scale=binning_scale,
        )
    
    rng = np.random.default_rng(42)
    gap_probability = .02  # probability of skipping an image
    
    for i in tqdm(range(n_images), desc="Generating observations", bar_format=bar_format):
        
        # randomly skip some images to create gaps
        if rng.random() < gap_probability:
            # if an image is skipped, increase the probability of skipping the next one to create larger gaps
            gap_probability = .95
            continue
        else:
            gap_probability = .02  # reset the probability of skipping the next image
        
        create_images(
            out_directory=out_directory,
            variable_source=variable_source,
            source_positions=source_positions,
            fluxes=fluxes,
            i=i,
            binning_scale=binning_scale,
            circular_aperture=circular_aperture,
            overwrite=overwrite,
            )