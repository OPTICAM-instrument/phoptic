from functools import partial
import json
from logging import Logger
from multiprocessing import cpu_count
import os
from pathlib import Path
from typing import Callable, Literal


from astroalign import find_transform
from astropy import units as u
from astropy.io.fits import Header
from astropy.table import QTable
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from photutils.segmentation import detect_threshold
from skimage.transform import matrix_transform, SimilarityTransform, warp
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


from phoptic.utils.transforms import find_translation
from phoptic.background.global_background import BaseBackground, DefaultBackground
from phoptic.correctors import BiasCorrector, DarkNoiseCorrector, FlatFieldCorrector
from phoptic.finders import DefaultFinder, get_source_coords_from_image
from phoptic.fitting.routines import fit_psf
from phoptic.instruments import Instrument, OPTICAM_MX
from phoptic.mef_slice import MEFSlice
from phoptic.noise import snr, snr_stderr, get_bias_stderr, get_dark_stderr, get_flat_stderr, get_read_stderr, \
    get_scint_stderr, get_shot_stderr, get_sky_stderr
from phoptic.photometers import AperturePhotometer, BasePhotometer
from phoptic.plotting.gifs import compile_gif, create_gif_frame
from phoptic.plotting.plots import plot_systematics, plot_background_meshes, plot_catalogs, plot_growth_curves, \
    plot_time_between_files, plot_psf, plot_rms_vs_median_flux, plot_noise, plot_snrs, plot_apertures
from phoptic.utils.batching import get_batches, get_batch_size
from phoptic.utils.constants import bar_format, counts_to_mag_factor
from phoptic.utils.data_checks import scan_data
from phoptic.utils.fits_handlers import get_data, get_stacked_images, save_stacked_images
from phoptic.utils.helpers import delete_keys_from_nested_dict, match_dict_keys
from phoptic.utils.logging import configure_logger, log_psf_params, recursive_log




class Reducer:
    """
    Class for reducing astronomical images.
    """


    def __init__(
        self,
        out_directory: Path | str,
        data_directory: Path | str,
        aperture_selector: Callable[[NDArray[np.float64]], NDArray[np.float64]] = np.median,
        background: BaseBackground | None = None,
        barycenter: bool = True,
        bias_corrector: BiasCorrector | None = None,
        dark_corrector: DarkNoiseCorrector | None = None,
        finder: None | Callable = None,
        flat_corrector: FlatFieldCorrector | None = None,
        instrument: Instrument = OPTICAM_MX(),
        number_of_processors: int = cpu_count() // 2,
        rebin_factor: int = 1,
        image_filter: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        remove_cosmic_rays: bool = False,
        show_plots: bool = True,
        threshold: float = 5,
        verbose: bool = True
        ) -> None:
        """
        Class for reducing astronomical images.
        
        Parameters
        ----------
        out_directory : Path | str
            The path to the directory to save the output files.
        data_directory : Path | str
            The path to the directory containing the data. Data may be single- and/or multi-extension FITS files.
        aperture_selector : Callable, optional
            The aperture selector, by default `np.median`. This function is used to select the aperture size for
            photometry. If a callable is provided, it should take a list of source sizes (`list[float]`) as input and
            return a single value.
        background : BaseBackground | None, optional
            The background calculator, by default `None`. If `None`, the default background calculator is used. If a
            callable is provided, it should take an image (`NDArray`) as input and return a `Background2D` object.
        barycenter : bool, optional
            Whether to apply a barycentric correction to the image timestamps, by default `True`.
        bias_corrector : BiasCorrector | None, optional
            The bias corrector, by default `None`. If `None`, no bias corrections are performed. See 
            https://opticam.readthedocs.io/en/latest/_executed/applying_corrections.html for more details.
        dark_corrector : DarkNoiseCorrector, optional,
            The dark noise corrector, by default `None`. If `None`, no dark noise corrections are performed. See
            https://opticam.readthedocs.io/en/latest/_executed/applying_corrections.html for more details.
        finder : Callable, optional
            The source finder, by default `None`. If `None`, the default source finder is used. If a callable is
            provided, it should take an image (`NDArray`) and a threshold (`float | NDArray`) as input and return a
            `QTable` instance. See https://opticam.readthedocs.io/en/latest/_executed/finders.html for details.
        flat_corrector : FlatFieldCorrector | None, optional,
            The flat-field corrector, by default `None`. If `None`, no flat-field corrections are performed. See 
            https://opticam.readthedocs.io/en/latest/_executed/applying_corrections.html for more details.
        instrument : Instrument, optional
            The instrument, by default `OPTICAM_MX()`. To use a custom instrument, see
            https://opticam.readthedocs.io/en/latest/_executed/instruments.html for details.
        number_of_processors : int, optional
            The number of processors to use for parallel processing, by default half the number of available processors.
        rebin_factor: int, optional
            The rebinning factor, by default 1 (no rebinning). The rebinning factor is the factor by which the image is
            rebinned in both dimensions. Rebinning can improve the detectability of faint sources and speed up
            some operations (like cosmic ray removal) at the cost of image resolution.
        image_filter : Callable[[NDArray[np.float64]], NDArray[np.float64]] | None, optional
            The kernel/filter to apply to images as they are opened, by default `None`. Paez+2026:
            https://ui.adsabs.harvard.edu/abs/2026RASTI...5ag021P/abstract found that a 3x3 median filter
            (e.g., `scipy.ndimage.median_filter()`) can be used to correct for warm pixels in long exposures (> 10 s)
            with OPTICAM.
        remove_cosmic_rays : bool, optional
            Whether to remove cosmic rays from images, by default `False`. Cosmic rays are removed using the LACosmic
            algorithm as implemented in `astroscrappy`. Note: this can be computationally expensive, particularly for
            large images.
        show_plots : bool, optional
            Whether to show plots as they're created, by default `True`. Whether `True` or `False`, plots are always
            saved to `out_directory`.
        threshold : float, optional
            The signal-to-noise ratio threshold for source finding, by default 5. Reduce this value to identify fainter
            sources, though this may lead to the identification of spurious sources.
        verbose : bool, optional
            Whether to print verbose output, by default `True`.
        """
        
        self.verbose = verbose
        
        ########################################### out_directory ###########################################
        
        self.out_directory = Path(out_directory)
        
        # create output directory if it does not exist
        if not self.out_directory.is_dir():
            if self.verbose:
                print(f"[PHOPTIC] {self.out_directory} not found, attempting to create ...")
            # create output directory if it does not exist
            try:
                os.makedirs(self.out_directory)
            except Exception as e:
                raise FileNotFoundError(f"[PHOPTIC] Could not create directory {self.out_directory} due to the following exception: {e}.")
            if self.verbose:
                print(f"[PHOPTIC] {self.out_directory} created.")
        
        
        ########################################### logger ###########################################
        
        self.logger = configure_logger(
            out_directory=self.out_directory,
            verbose=self.verbose,
            )
        
        ########################################### sub-directories ###########################################
        
        # create subdirectories
        if not self.out_directory.joinpath("cat").is_dir():
            os.makedirs(self.out_directory.joinpath("cat"))
        if not self.out_directory.joinpath("diag").is_dir():
            os.makedirs(self.out_directory.joinpath("diag"))
        if not self.out_directory.joinpath("misc").is_dir():
            os.makedirs(self.out_directory.joinpath("misc"))
        
        ########################################### input params ###########################################
        
        self.data_directory = Path(data_directory)
        self.rebin_factor = rebin_factor
        self.image_filter = image_filter
        self.instrument = instrument
        self.aperture_selector = aperture_selector
        self.threshold = threshold
        self.remove_cosmic_rays = remove_cosmic_rays
        self.barycenter = barycenter
        self.number_of_processors = number_of_processors
        self.show_plots = show_plots
        
        if self.barycenter:
            self.time_key = 'BMJD'
        else:
            self.time_key = 'MJD'
        
        ########################################### check input data ###########################################
        
        self.camera_files, self.binning_scale, self.bmjds, self.ignored_files, self.t_ref = scan_data(
                data_directory=self.data_directory,
                out_directory=self.out_directory,
                instrument=self.instrument,
                barycenter=self.barycenter,
                verbose=self.verbose,
                return_output=True,
                logger=self.logger,
                number_of_processors=self.number_of_processors,
                )
        
        ########################################### correctors ###########################################
        
        self.bias_corrector = bias_corrector
        if self.bias_corrector is not None:
            errors = self.bias_corrector.run_checks(
                data_files_by_key=self.camera_files,
                return_errors=True,
                )
            if errors == 1:
                raise ValueError(f'[PHOPTIC] {errors} BiasCorrector error needs to be resolved.')
            elif errors > 1:
                raise ValueError(f'[PHOPTIC] {errors} BiasCorrector errors need to be resolved.')
        
        self.dark_corrector = dark_corrector
        if self.dark_corrector is not None:
            errors = self.dark_corrector.run_checks(
                data_files_by_key=self.camera_files,
                return_errors=True,
                )
            if errors == 1:
                raise ValueError(f'[PHOPTIC] {errors} DarkNoiseCorrector error needs to be resolved.')
            elif errors > 1:
                raise ValueError(f'[PHOPTIC] {errors} DarkNoiseCorrector errors need to be resolved.')
        
        self.flat_corrector = flat_corrector
        if self.flat_corrector is not None:
            errors = self.flat_corrector.run_checks(
                data_files_by_key=self.camera_files,
                return_errors=True,
                )
            if errors == 1:
                raise ValueError(f'[PHOPTIC] {errors} FlatFieldCorrector error needs to be resolved.')
            elif errors > 1:
                raise ValueError(f'[PHOPTIC] {errors} FlatFieldCorrector errors need to be resolved.')
        
        ########################################### plot time between files ###########################################
        
        plot_time_between_files(
            out_directory=self.out_directory,
            camera_files=self.camera_files,
            bmjds=self.bmjds,
            show=self.show_plots,
            save=True,
            )
        
        ########################################### define reference images ###########################################
        
        # define middle image as reference image for each filter
        reference_indices: dict[str, int] = {}
        self.reference_files: dict[str, MEFSlice] = {}
        for key in self.camera_files.keys():
            reference_indices[key] = len(self.camera_files[key]) // 2
            self.reference_files[key] = self.camera_files[key][reference_indices[key]]
        
        ########################################### aperture selector ###########################################
        
        assert callable(aperture_selector), "[PHOPTIC] aperture_selector must be callable."
        self.aperture_selector = aperture_selector
        
        ########################################### background ###########################################
        
        if background is None:
            box_size = 2048 // self.binning_scale // self.rebin_factor // 32
            self.background = DefaultBackground(box_size)
            self.logger.debug(f'Using default background estimator with box_size={box_size}.')
        elif callable(background):
            # use custom background estimator
            self.background = background
            self.logger.debug(f'Using custom background estimator {background.__class__.__name__} with parameters {background.__dict__}.')
        else:
            raise ValueError('[PHOPTIC] background must be a callable or None. If None, the default background estimator is used.')
        
        ########################################### finder ###########################################
        
        if finder is None:
            effective_image_size = 2048 // self.binning_scale // self.rebin_factor
            n_pixels = 128 // (2048 // effective_image_size)**2
            border_width = 2048 // self.binning_scale // self.rebin_factor // 16
            self.finder = DefaultFinder(n_pixels, border_width)
            self.logger.debug(f'Using default source finder with n_pixels={n_pixels} and border_width={border_width}.')
        elif callable(finder):
            self.finder = finder
            self.logger.debug(f'Using custom source finder {finder.__class__.__name__} with parameters {finder.__dict__}.')
        else:
            raise ValueError('[PHOPTIC] finder must be a callable or None. If None, the default source finder is used.')
        
        ########################################### log input params ###########################################
        
        self._log_params()
        
        ####################################### define convenience methods #######################################
        
        self.get_data: Callable[[MEFSlice], tuple[NDArray[np.float64], Header, dict[str, float | NDArray[np.float64]]]] = partial(
            get_data,
            instrument=self.instrument,
            rebin_factor=self.rebin_factor,
            image_filter=self.image_filter,
            remove_cosmic_rays=self.remove_cosmic_rays,
            bias_corrector=self.bias_corrector,
            dark_corrector=self.dark_corrector,
            flat_corrector=self.flat_corrector,
            )
        
        ########################################### misc attributes ###########################################
        
        self.transforms = {}  # define transforms as empty dictionary
        self.unaligned_files = []  # define unaligned files as empty list
        self.catalogs : dict[str, QTable] = {}  # define catalogs as empty dictionary
        self.psf_params = {}  # define PSF parameters as empty dictionary
        
        ########################################### read transforms ###########################################
        
        if os.path.isfile(os.path.join(self.out_directory, "cat/transforms.json")):
            with open(os.path.join(self.out_directory, "cat/transforms.json"), "r") as file:
                self.transforms.update(json.load(file))
            
            if self.verbose:
                self.logger.info("Read transforms from file.")
        
        ########################################### read catalogs ###########################################
        
        for key in list(self.camera_files.keys()):
            file_path = os.path.join(self.out_directory, f"cat/{key}_catalog.ecsv")
            if os.path.isfile(file_path):
                self.catalogs.update(
                    {
                        key: QTable.read(
                            file_path,
                            format="ascii.ecsv",
                            )
                        }
                    )
                self.psf_params[key] = set_psf_params(
                    aperture_selector=self.aperture_selector,
                    catalog=self.catalogs[key],
                    )
                
                self.logger.info(f"Read {key} catalog from file.")
        
        ########################################### read unaligned files ###########################################
        
        file_path = os.path.join(self.out_directory, 'diag/unaligned_files.txt')
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                for line in file:
                    self.unaligned_files.append(line)
            
            self.logger.info(f"Read unaligned files from file.")


    def _log_params(self) -> None:
        """
        Log the input parameters of a `Reducer` instance to file.
        
        Parameters
        ----------
        reducer : Reducer
            The `Reducer` instance.
        """
        
        # get parameters
        params = dict(recursive_log(self))
        
        params.update({'keys': list(self.camera_files.keys())})
        
        keys_to_remove: set[str] = {
            'bmjds',  # logged separately
            'catalogs',  # logged separately
            'logger',  # logged separately
            'master_images',  # unnecessary
            'master_variances',  # unnecessary
            'number_of_processors',  # unnecessary
            'passed_checks',  # unnecessary
            'transforms',  # logged separately
            'unaligned_files',  # logged separately
            'verbose',  # unnecessary
            }
        delete_keys_from_nested_dict(d=params, keys=keys_to_remove)
        
        # sort parameters
        params = dict(sorted(params.items()))
        
        save_path = self.out_directory / 'misc' / 'reduction_parameters.json'
        if save_path.is_file():
            # get existing params
            with open(save_path, 'r') as file:
                file_params = json.load(file)
            
            # check params match
            if json.dumps(file_params, sort_keys=True) != json.dumps(params, sort_keys=True):
                raise ValueError(f'[PHOPTIC] Cannot instantiate Reducer: incompatible reduction_parameters.json file found in out_directory/misc. Consider deleting the contents of out_directory to start from scratch, or instantiate the Reducer with the same parameters as those listed in the existing reduction_parameters.json file.')
        else:
            # only write params to file if the file doesn't already exist
            with open(save_path, "w") as file:
                json.dump(params, file, indent=4)


    def create_catalogs(
        self,
        max_catalog_sources: int = 15,
        n_alignment_sources: int = 15,
        transform_type: Literal['affine', 'translation'] = 'affine',
        rotation_limit: float | None = None,
        translation_limit: float | int | list[float | int] | None = None,
        scale_limit: float | None = None,
        overwrite: bool = False,
        ) -> None:
        """
        Initialise the source catalogs for each camera. Some aspects of this method are parallelised for speed.
        
        Parameters
        ----------
        max_catalog_sources : int, optional
            The maximum number of sources to include in the catalog, by default 30. Since source IDs are ordered by
            brightness, the brightest `max_catalog_sources` sources are included in the catalog.
        n_alignment_sources : int, optional
            The (maximum) number of sources to use for image alignment, by default 30. If
            `transform_type='translation'`, `n_alignment_sources` must be >= 1, and the brightest `n_alignment_sources`
            sources are used for image alignment. If `transform_type='affine'`, `n_alignment_sources` must be >= 3 and
            represents that *maximum* number of sources that *may* be used for image alignment.
        transform_type : Literal['affine', 'translation'], optional
            The type of transform to use for image alignment, by default 'affine'. 'translation' performs simple
            x, y translations, while 'affine' uses `astroalign.find_transform()`. 'affine' is generally more robust 
            (and is therefore the default) while 'translation' can work with fewer sources.
        rotation_limit : float, optional
            The maximum rotation limit (in degrees) for affine transformations, by default `None` (no limit).
        scale_limit : float, optional
            The maximum scale limit for affine transformations, by default `None` (no limit).
        translation_limit : float | int | list[float | int] | None, optional
            The maximum translation limit for both types of transformations, by default `None` (no limit). Can be a
            scalar value that applies to both x- and y-translations, or an iterable where the first value defines the
            x-translation limit and the second value defines the y-translation limit.
        overwrite : bool, optional
            Whether to overwrite existing catalogs, by default False.
        """
        
        assert transform_type in ['affine', 'translation'], '[PHOPTIC] transform_type must be either "affine" or "translation".'
        
        if translation_limit is not None:
            # if a scalar translation limit is specified, convert it to a list
            if isinstance(translation_limit, float) or isinstance(translation_limit, int):
                translation_limit = [translation_limit, translation_limit]
        
        # if catalogs already exist, skip
        if os.path.isfile(os.path.join(self.out_directory, 'cat/catalogs.pdf')) and not overwrite:
            self.logger.info('Catalogs already exist. To overwrite, set overwrite=True.')
            
            plot_catalogs(
                out_directory=self.out_directory,
                stacked_images=get_stacked_images(self.out_directory),
                catalogs=self.catalogs,
                show=self.show_plots,
                save=False,
            )
            
            return
        
        self.logger.info('Creating source catalogs.')
        
        stacked_images = {}
        
        # for each camera
        for key in self.camera_files.keys():
            
            # if no images found for camera, skip
            if len(self.camera_files[key]) == 0:
                continue
            
            # get reference image
            reference_image = get_data(
                file=self.reference_files[key],
                instrument=self.instrument,
                bias_corrector=self.bias_corrector,
                dark_corrector=self.dark_corrector,
                flat_corrector=self.flat_corrector,
                rebin_factor=self.rebin_factor,
                image_filter=self.image_filter,
                remove_cosmic_rays=self.remove_cosmic_rays,
                )[0]
            
            try:
                # get source coordinates in descending order of brightness
                reference_coords = get_source_coords_from_image(
                    reference_image,
                    finder=self.finder,  # type: ignore
                    threshold=self.threshold,
                    background=self.background,
                    n_sources=n_alignment_sources,
                    )
            except Exception as e:
                self.logger.error(f'No sources detected in {key} reference image ({self.reference_files[key]}): {e}. Reducing threshold or n_pixels in the source finder may help.')
                continue
            
            if len(reference_coords) < n_alignment_sources and transform_type == 'translation':
                self.logger.error(f'Found {len(reference_coords)} sources in {key} reference image ({self.reference_files[key]}) but n_alignment_sources={n_alignment_sources}. transform_type="translation" requires at least n_alignment_sources be detected in the reference image to work. Consider reducing n_alignment_sources and/or threshold, or using transform_type="affine".')
                continue
            
            self.logger.debug(f'{key} alignment source coordinates: {reference_coords}')
            
            # align and stack images in batches
            batches = get_batches(self.camera_files[key])
            results = process_map(
                partial(
                    self._align_batch,
                    reference_image_shape=reference_image.shape,
                    reference_coords=reference_coords,
                    transform_type=transform_type,
                    rotation_limit=rotation_limit,
                    scale_limit=scale_limit,
                    translation_limit=translation_limit,
                    n_alignment_sources=n_alignment_sources,
                    ),
                batches,
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f'[PHOPTIC] Aligning {key} images',
                bar_format=bar_format,
                tqdm_class=tqdm,
                )
            
            self.transforms, self.unaligned_files, stacked_image, systematics = parse_alignment_results(
                results=results,
                camera_files=self.camera_files[key],
                transforms=self.transforms,
                unaligned_files=self.unaligned_files,
                logger=self.logger,
                )
            
            # estimate threshold for source detection
            threshold = detect_threshold(
                stacked_image,
                n_sigma=self.threshold,
                )
            
            try:
                # identify sources in stacked image
                cat = self.finder(
                    stacked_image,
                    threshold,
                    )
            except Exception as e:
                self.logger.error(f'No sources detected in the stacked {key} stacked image: {e}. Reducing threshold may help.')
                continue
            
            # save stacked image
            stacked_images[key] = stacked_image
            
            # limit catalog to brightest sources
            cat = cat[:max_catalog_sources]
            
            # update catalogs
            self.catalogs.update({key: cat})  # type: ignore
            
            # save updated catalogs
            save_catalog(
                catalog=cat,
                key=key,
                out_directory=self.out_directory,
                )
            
            self.psf_params[key] = set_psf_params(
                aperture_selector=self.aperture_selector,
                catalog=self.catalogs[key],
                )
            
            self.save_systematics(
                systematics=systematics,
                key=key,
                )
        
        log_psf_params(
            out_directory=self.out_directory,
            psf_params=self.psf_params,
            binning_scale=self.binning_scale,
            rebin_factor=self.rebin_factor,
            pixel_scales=self.instrument.pixel_scales
            )
        
        plot_catalogs(
            out_directory=self.out_directory,
            stacked_images=stacked_images,
            catalogs=self.catalogs,
            show=self.show_plots,
            save=True,
            )
        
        save_stacked_images(
            stacked_images=stacked_images,
            out_directory=self.out_directory,
            overwrite=overwrite,
            )
        
        plot_systematics(
            out_directory=self.out_directory,
            instrument=self.instrument,
            bin_factor=self.binning_scale * self.rebin_factor,
            t_ref=self.t_ref,
            time_key=self.time_key,
            show=self.show_plots,
            save=True,
            )
        
        # save transforms to file
        if not os.path.isfile(os.path.join(self.out_directory, "cat/transforms.json")) or overwrite:
            with open(os.path.join(self.out_directory, "cat/transforms.json"), "w") as file:
                json.dump(self.transforms, file, indent=4)
        
        # save unaligned files to file
        save_unaligned_files(
            out_directory=self.out_directory,
            unaligned_files=self.unaligned_files,
            )


    def pick_sources(
        self,
        percentile: float = 99.0,
        region_size: int | None = None,
        ):
        """
        Interactive plot for manually adding sources to the catalogs.
        
        Parameters
        ----------
        percentile: float, optional
            The interval percentile for image normalisation, by default `99.0`.
        region_size : int | None, optional
            The size of the region used to refine source coordinates, by default `None`. If `None`, the region size is set to the image width divided by 32.
        
        Returns
        -------
        QTable
            The table containing the new
        """
        
        print(f'[PHOPTIC] Warning: Reducer.add_sources() requires a compatible matplotlib backend.\n\
            You may need to add the line, e.g., "%matploblib widget" before calling add_sources().')
        
        stacked_images = get_stacked_images(self.out_directory)
        
        fig = plot_catalogs(
            out_directory=self.out_directory,
            stacked_images=stacked_images,
            catalogs=self.catalogs,
            show=False,
            save=False,
            percentile=percentile,
            return_fig=True,
            )
        
        if region_size is None:
            region_size = max(list(stacked_images.values())[0].shape) // 64
        
        add_source = partial(
            self._add_source,
            fig=fig,
            stacked_images=stacked_images,
            region_size=region_size,
            )
        
        fig.canvas.mpl_connect('button_press_event', add_source)
        
        plt.show()


    def _add_source(
        self,
        event: Literal['button_press_event'],
        fig: Figure,
        stacked_images: dict[str, NDArray],
        region_size: int,
        ) -> None:
        """
        Add a manually-picked source to the associated catalog.
        
        Parameters
        ----------
        event : Literal["button_press_event"]
            The event.
        fig : Figure
            The catalog plot.
        stacked_images : dict[str, NDArray]
            The catalog images.
        region_size : int | None, optional
            The size of the region used to refine source coordinates.
        """
        
        if event.inaxes is None:
            return
        
        ax = event.inaxes
        key = ax.get_title()  # get camera:filter key
        
        # get clicked coordinates
        x, y = event.xdata, event.ydata
        int_x, int_y = int(x), int(y)
        
        # get a small region around the clicked coordinates
        region = stacked_images[key][int_y-region_size:int_y+region_size, int_x-region_size:int_x+region_size]
        
        # get peak flux coordinates in region
        # this is the initial guess for our source position
        y_init, x_init = np.unravel_index(np.argmax(region), region.shape)
        
        # get PSF coordinates in region
        x_region, y_region, theta_rad = fit_psf(
            image=region,
            x_init=int(x_init),
            y_init=int(y_init),
            semimajor_axis=self.psf_params[key]['semimajor_axis'],
            semiminor_axis=self.psf_params[key]['semiminor_axis'],
            )
        
        # convert region coordinates to image coordinates
        x_image = x_region + x - region_size
        y_image = y_region + y - region_size
        
        # add minimally required information for chosen source to catalog
        new_row = {
            'x_centroid': x_image,
            'y_centroid': y_image,
            'orientation': np.rad2deg(theta_rad) * u.deg,
            }
        self.catalogs[key].add_row(new_row)
        
        save_catalog(
            catalog=self.catalogs[key],
            key=key,
            out_directory=self.out_directory,
            )
        
        patch = Circle(
            (x_image, y_image),
            region_size,
            edgecolor='r',
            facecolor='none',
            )
        ax.add_patch(patch)
        fig.canvas.draw()


    def _align_batch(
        self,
        batch: list[MEFSlice],
        reference_image_shape: tuple[int],
        reference_coords: NDArray,
        transform_type: Literal['affine', 'translation'],
        rotation_limit: float | None,
        scale_limit: float | None,
        translation_limit: list[float] | None,
        n_alignment_sources: int,
        ) -> tuple[
            NDArray[np.float64],
            dict[str, list[float]],
            dict[str, dict[str, float]],
            list[tuple[str, str]],
            ]:
        """
        Align a batch of images with respect to some reference coordinates.
        
        Parameters
        ----------
        batch: list[MEFSlice]
            The files.
        reference_image_shape : tuple[int]
            The reference image's shape.
        reference_coords : NDArray
            The source coordinates in the reference image.
        transform_type : Literal['affine', 'translation']
            The type of transform to use for image alignment.
        rotation_limit : float | None
            The maximum rotation limit (in degrees) for image alignment.
        scale_limit : float | None
            The maximum scaling limit for image alignment.
        translation_limit : list[float] | None
            The maximum translation limit for image alignment.
        n_alignment_sources : int
            The (maximum) number of sources to use for image alignment.
        
        Returns
        -------
        tuple[NDArray[np.float64], dict[str, list[float]], dict[str, dict[str, float]], list[tuple[str, str]]]
            The stacked image, transforms, systematics, and log messages.
        """
        
        stacked_image = np.zeros(reference_image_shape)  # create empty stacked image
        transforms: dict[str, list[float]] = {}
        systematics_dict: dict[str, dict[str, float]] = {}
        queued_logs: list[tuple[str, str]] = []
        
        for file in batch:
            data, header, noise_dict = get_data(
                file=file,
                instrument=self.instrument,
                bias_corrector=self.bias_corrector,
                dark_corrector=self.dark_corrector,
                flat_corrector=self.flat_corrector,
                rebin_factor=self.rebin_factor,
                image_filter=self.image_filter,
                remove_cosmic_rays=self.remove_cosmic_rays,
                )
            
            # calculate and subtract background
            bkg = self.background(data)
            
            # identify sources
            try:
                coords, fwhm = get_source_coords_from_image(
                    data,
                    finder=self.finder,  # type: ignore
                    threshold=self.threshold,
                    bkg=bkg,
                    return_fwhm=True,
                    aperture_selector=self.aperture_selector,
                    )
            except Exception as e:
                queued_logs.append(('error', f'No sources detected in {file.path} extension {file.ext}: {e} Skipping.'))
                continue
            
            if len(coords) < n_alignment_sources and transform_type == 'translation':
                queued_logs.append(('error', f'{len(coords)} sources detected in {file.path} extension {file.ext} but n_alignment_sources={n_alignment_sources} and transform_type="translation". Skipping. To attempt to align images in which fewer than n_alignment_sources are detected, try transform_type="affine".'))
                continue
            
            if transform_type == 'translation':
                # find translation
                transform = find_translation(
                    coords,
                    reference_coords,
                    )
            else:
                try:
                    # find affine transformation using astroalign
                    transform = find_transform(
                        reference_coords,
                        coords,
                        max_control_points=n_alignment_sources,
                        )[0]
                except Exception as e:
                    queued_logs.append(('error', f'Could not align {file.path} extension {file.ext} due to the following exception: {e}. Skipping.'))
                    continue
            
            # validate transform
            valid_transform, log_message = self._valid_transform(
                file=file,
                transform=transform,
                rotation_limit=rotation_limit,
                scale_limit=scale_limit,
                translation_limit=translation_limit,
                )
            if not valid_transform:
                queued_logs.append(log_message)
                continue
            
            transforms[file.key] = transform.params.tolist()
            systematics_dict[file.key] = {
                'bkg_median': bkg.background_median,
                'bkg_rms': bkg.background_rms_median,
                'FWHM': fwhm,
                'airmass': self.instrument.get_airmass(header=header),
                'rel_scint_noise': noise_dict['rel_scint_noise'],
                }
            
            # transform and stack image
            stacked_image += warp(
                data - bkg.background,
                transform,
                output_shape=reference_image_shape,
                order=3,
                mode='constant',
                cval=0.,
                clip=True,
                preserve_range=True,
                )
        
        return stacked_image, transforms, systematics_dict, queued_logs


    def _valid_transform(
        self,
        file: MEFSlice,
        transform: SimilarityTransform,
        rotation_limit: float | None,
        scale_limit: float | None,
        translation_limit: list[float] | None,
        ) -> tuple[bool, None | tuple[str, str]]:
        """
        Find whether a transform is valid given some transform limits.
        
        Parameters
        ----------
        file : MEFSlice
            The path to the file being transformed.
        transform : SimilarityTransform
            The transform.
        rotation_limit : float | None
            The rotation limit.
        scale_limit : float | None
            The scale limit.
        translation_limit : list[float] | None
            The translation limit.
        
        Returns
        -------
        tuple[bool, None | tuple[str, str]]
            Whether the transform is valid. If not, a log message is also returned as a tuple: (log level, log string).
        """
        
        if rotation_limit:
            if abs(transform.rotation) > rotation_limit:
                return False, ('error', f'File {file.path} extension {file.ext} transform exceeded rotation limit. Rotation limit is {rotation_limit}, but rotation was {transform.rotation}.')
        if scale_limit:
            if transform.scale > scale_limit:
                return False, ('error', f'File {file.path} extension {file.ext} transform exceeded scale limit. Scale limit is {scale_limit}, but scale was {transform.scale}.')
        if translation_limit:
            if abs(transform.translation[0]) > translation_limit[0] or abs(transform.translation[1]) > translation_limit[1]:
                return False, ('error', f'File {file.path} extension {file.ext} transform exceeded translation limit. Translation limit is {translation_limit}, but translation was {transform.translation}.')
        
        return True, None


    def save_systematics(
        self,
        systematics: dict[str, dict[str, float]],
        key: str,
        ) -> None:
        """
        Save the systematics to a CSV file.
        
        Parameters
        ----------
        systematics : dict[Path, dict[str, float]]
            The systematics for each file.
        key : str
            The camera:filter key.
        """
        
        df = pd.DataFrame.from_dict(systematics, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'file'}, inplace=True)
        
        time_df = pd.DataFrame.from_dict(self.bmjds, orient='index')
        time_df.reset_index(inplace=True)
        time_df.columns = ['file', self.time_key]
        time_df.rename(columns={'index': 'file'}, inplace=True)
        
        merged_df = pd.merge(time_df, df, on='file', how='inner')  # merge dataframes to get corresponding times
        merged_df.drop('file', axis=1, inplace=True)  # delete file column
        merged_df.to_csv(os.path.join(self.out_directory, f'diag/{key}_systematics.csv'), index=False)


    def plot_background_meshes(
        self,
        save: bool = False,
        ) -> None:
        """
        Plot the background mesh over an image from each filter to verify it's appropriately sized. If stacked catalog
        images exist, those will be used. Otherwise, a random image will be chosen for each filter.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot, by default `False`.
        """
        
        catalogs = True
        for key in self.camera_files.keys():
            if key not in self.catalogs.keys():
                catalogs = False
        
        if catalogs:
            images = get_stacked_images(self.out_directory)
        else:
            images = self.get_random_image_for_each_filter()
            
            # subtract background
            for label, image in images.items():
                bkg = self.background(image)
                images[label] = image - bkg.background
        
        plot_background_meshes(
            out_directory=self.out_directory,
            images=images,
            background=self.background,
            show=self.show_plots,
            save=save,
            )


    def get_random_image_for_each_filter(
        self,
        ) -> dict[str, NDArray]:
        """
        Choose a random image for each filter from a dictionary.
        
        Returns
        -------
        dict[str, NDArray]
            A dictionary containing a random image for each filter
        """
        
        rng = np.random.default_rng()
        images = {}
        
        for files in self.camera_files.values():
            file = files[rng.choice(len(files))]  # choose a random file
            key = f'{file.path.name} ext {file.ext}'
            images[key] = self.get_data(file)[0]
        
        return images


    def plot_growth_curves(
        self,
        targets: dict[str, int | list[int]] | None = None,
        save: bool = False,
        ) -> None:
        """
        Plot the growth curves for the sources identified in the catalog images. The resulting plots are saved to
        out_directory/diag/growth_curves as PDF files.
        
        Parameters
        ----------
        targets : dict[str, int | list[int]] | None, optional
            The targets for which growth curves will be created, by default `None` (growth curves are created for all
            catalog sources). To create growth curves for specific targets, pass a dictionary with keys listing the
            desired filters and values listing each filter's correpsonding target(s). For example:
            ```
            # plot growth curves for the three brightest sources in each catalog
            plot_growth_curves(
                targets = {
                    'g': [1, 2, 3],
                    'r': [1, 2, 3],
                    'i': [1, 2, 3],
                    },
                )
            ```
        save : bool, optional
            Whether to save the plots, by default `False`.
        """
        
        stacked_images = get_stacked_images(self.out_directory)
        
        # create targets dict if it does not already exist
        if targets is None:
            growth_curve_targets = create_targets_dict(self.catalogs)
        else:
            growth_curve_targets = targets
        
        self.logger.info(f'Generating growth curves for targets: {repr(growth_curve_targets)}')
        
        for key, cat in self.catalogs.items():
            
            if key not in growth_curve_targets.keys():
                self.logger.error(f'Key {key} is not in target dictionary. Skipping.')
                continue
            
            fig = plot_growth_curves(
                image=stacked_images[key],
                cat=cat,
                targets=growth_curve_targets[key],
                psf_params=self.psf_params[key],
                )
            
            fig.suptitle(key, fontsize='large')
            
            dir_path = os.path.join(self.out_directory, 'diag/growth_curves')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            
            if save:
                fig.savefig(os.path.join(dir_path, f'{key}_growth_curves.pdf'))
            
            if self.show_plots:
                plt.show(fig)
            else:
                plt.close(fig)
        
        self.logger.info('Growth curves generated.')


    def plot_psfs(
        self,
        ) -> None:
        """
        Plot the PSFs for the catalog sources.
        """
        
        if not os.path.isdir(os.path.join(self.out_directory, 'psfs')):
            os.makedirs(os.path.join(self.out_directory, 'psfs'))
        
        # get stacked images
        stacked_images = get_stacked_images(self.out_directory)
        
        for key in self.catalogs.keys():
            
            a = self.psf_params[key]['semimajor_axis']
            b = self.psf_params[key]['semiminor_axis']
            
            for source_indx in range(len(self.catalogs[key])):
                plot_psf(
                    catalog=self.catalogs[key],
                    source_indx=source_indx,
                    stacked_image=stacked_images[key],
                    key=key,
                    a=a,
                    b=b,
                    out_directory=self.out_directory,
                )


    def plot_snrs(
        self,
        save: bool = False,
        ) -> None:
        """
        Plot the signal-to-noise ratios for each catalogued source in the reference images.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot, by default `False`.
        """
        
        snrs: dict[str, dict[int, float]] = {}
        for key in self.camera_files.keys():
            snrs[key] = self.get_snrs(
                file=self.reference_files[key],
                key=key,
                )
        
        plot_snrs(
            out_directory=self.out_directory,
            snrs=snrs,
            show=self.show_plots,
            save=save,
        )


    def create_gifs(
        self,
        keep_frames: bool = True,
        overwrite: bool = False,
        ) -> None:
        """
        Create alignment gifs for each camera. Some aspects of this method are parallelised for speed. The frames are 
        saved in out_directory/diag/*_gif_frames and the GIFs are saved in out_directory/cat.
        
        Parameters
        ----------
        keep_frames : bool, optional
            Whether to save the GIF frames in out_directory/diag, by default True. If False, the frames will be deleted
            after the GIF is saved.
        overwrite : bool, optional
            Whether to overwrite existing GIFs, by default False.
        """
        
        # for each camera
        for key in list(self.catalogs.keys()):
            
            # skip cameras with no images
            if len(self.camera_files[key]) == 0:
                continue
            elif os.path.exists(os.path.join(self.out_directory, f"cat/{key}_images.gif")) and not overwrite:
                self.logger.info(f"{key} GIF already exists. To overwrite, set overwrite to True.")
                continue
            
            # create gif frames directory if it does not exist
            if not os.path.isdir(os.path.join(self.out_directory, f"diag/{key}_gif_frames")):
                os.mkdir(os.path.join(self.out_directory, f"diag/{key}_gif_frames"))
            
            chunksize = get_batch_size(len(self.camera_files[key]))
            process_map(
                partial(
                    create_gif_frame,
                    out_directory=self.out_directory,
                    aperture_selector=self.aperture_selector,
                    catalog=self.catalogs[key],
                    key=key,
                    transforms=self.transforms,
                    reference_file=self.reference_files[key],
                    rebin_factor=self.rebin_factor,
                    image_filter=self.image_filter,
                    background=self.background,
                    instrument=self.instrument,
                    ),
                self.camera_files[key],
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f"[PHOPTIC] Creating {key} GIF frames",
                chunksize=chunksize,
                bar_format=bar_format,
                tqdm_class=tqdm,
                )
            
            # save GIF
            compile_gif(
                out_directory=self.out_directory,
                key=key,
                camera_files=self.camera_files,
                keep_frames=keep_frames,
                )


    def plot_apertures(
        self,
        photometer: AperturePhotometer,
        targets: dict[str, int] | dict[str, list[int]] | dict[str, list[int] | int] | None = None,
        save: bool = False,
        ) -> None:
        """
        Plot the apertures over each source.
        
        Parameters
        ----------
        photometer : AperturePhotometer
            The `AperturePhotometer` instance. If a local background estimator has been defined, this will also be
            plotted.
        targets : dict[str, int] | dict[str, list[int]] | dict[str, list[int] | int] | None
            The targets for which apertures will be plotted, by default `None` (apertures are plotted for all
            sources). To plot apertures for specific targets, pass a dictionary with keys listing the
            desired filters and values listing each filter's correpsonding target(s). For example:
            ```
            # plot apertures for the three brightest sources in each filter
            photometer = opticam.AperturePhotometer()
            plot_apertures(
                photometer=photometer,
                targets = {
                    'g': [1, 2, 3],
                    'r': [1, 2, 3],
                    'i': [1, 2, 3],
                    },
                )
            ```
        save : bool, optional
            Whether to save the plots, by default `False`.
        """
        
        if targets is None:
            targets = create_targets_dict(self.catalogs)
        
        for key in self.catalogs.keys():
            if key not in targets.keys():
                continue
            
            img = get_data(
                file=self.reference_files[key],
                instrument=self.instrument,
                bias_corrector=self.bias_corrector,
                dark_corrector=self.dark_corrector,
                flat_corrector=self.flat_corrector,
                rebin_factor=self.rebin_factor,
                image_filter=self.image_filter,
                remove_cosmic_rays=self.remove_cosmic_rays,
                )[0]
            
            plot_apertures(
                out_directory=self.out_directory,
                data=img,
                cat=self.catalogs[key],
                targets=targets[key],
                photometer=photometer,
                psf_params=self.psf_params[key],
                key=key,
                show=self.show_plots,
                save=save,
            )


    def photometry(
        self,
        photometer: BasePhotometer,
        overwrite: bool = False,
        ) -> None:
        """
        Perform photometry on the catalogs using the provided photometer.
        
        Parameters
        ----------
        photometer : BasePhotometer
            The photometer. Should be a subclass of `BasePhotometer`, or implement a `compute` method that follows the
            `BasePhotometer` interface.
        overwrite : bool, optional
            Whether to overwrite any existing light curves files computed using the same photometer, by default `False`.
        """
        
        # define save directory using the photometer name
        save_name = photometer.get_label()
        
        self.logger.info(f'Photometry results will be saved to lcs/{save_name} in {self.out_directory}.')
        
        save_dir = self.out_directory.joinpath(f"lcs/{save_name}")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # for each filter
        for key in self.catalogs.keys():
            if os.path.isfile(os.path.join(save_dir, f'{key}_source_1.csv')) and not overwrite:
                self.logger.info(f'Skipping {key} since existing light curves files were found. To overwrite these files, set overwrite=True.')
                continue
            
            cat_coords = np.array([self.catalogs[key]["x_centroid"].value,  # type: ignore
                                      self.catalogs[key]["y_centroid"].value],  # type:ignore
                                     ).T
            
            files = [file for file in self.camera_files[key] if file not in self.unaligned_files]
            batch_size = get_batch_size(len(files))
            results = process_map(
                partial(
                    self._perform_photometry,
                    photometer=photometer,
                    cat_coords=cat_coords,
                    key=key,
                ),
                files,
                max_workers=self.number_of_processors,
                disable=not self.verbose,
                desc=f"[PHOPTIC] Performing photometry on {key} images",
                chunksize=batch_size,
                bar_format=bar_format,
                tqdm_class=tqdm,
            )
            
            save_photometry_results(
                results=results,
                catalogs=self.catalogs,
                barycenter=self.barycenter,
                save_dir=save_dir,
                key=key
            )
        
        plot_rms_vs_median_flux(
            lc_dir=save_dir,
            save_dir=self.out_directory.joinpath('diag'),
            phot_label=save_name,
            show=self.show_plots,
            )


    def _perform_photometry(
        self,
        file: MEFSlice,
        photometer: BasePhotometer,
        cat_coords: NDArray,
        key: str,
        ) -> dict[str, list]:
        """
        Perform photometry on a file.
        
        Parameters
        ----------
        file : MEFSlice
            The file.
        photometer : BasePhotometer
            The photometer to use.
        cat_coords : NDArray
            The coordinates of the sources in the catalog.
        key : str
            The camera:filter key.
        
        Returns
        -------
        dict[str, list]
            The photometry results.
        """
        
        image, header, noise_dict = get_data(
            file=file,
            instrument=self.instrument,
            bias_corrector=self.bias_corrector,
            dark_corrector=self.dark_corrector,
            flat_corrector=self.flat_corrector,
            rebin_factor=self.rebin_factor,
            image_filter=self.image_filter,
            remove_cosmic_rays=self.remove_cosmic_rays,
            )
        
        if photometer.local_background_estimator is None:
            bkg = self.background(image)  # get 2D background
            image -= bkg.background  # remove background from image
            threshold = self.threshold * bkg.background_rms  # define source detection threshold
            background_rms = bkg.background_rms.copy()
        else:
            # estimate source detection threshold from noisy image
            threshold = detect_threshold(image, self.threshold)
            background_rms = None
        
        image_coords = None  # assume no image coordinates by default
        if not photometer.forced:
            tbl = self.finder(image, threshold)
            image_coords = np.array([
                tbl["x_centroid"].value,
                tbl["y_centroid"].value,
                ]).T
        
        # apply transform to catalogue coordinates
        transformed_cat_coords = matrix_transform(cat_coords, self.transforms[file.key])
        
        results = photometer.compute(
            image=image,
            bias_var=noise_dict['bias_var'],
            dark_var=noise_dict['dark_var'],
            flat_var=noise_dict['flat_var'],
            background_rms=background_rms,
            cat_coords=transformed_cat_coords,
            image_coords=image_coords,
            psf_params=self.psf_params[key],
            read_noise=self.instrument.get_read_noise(file=file),
            rel_scint_noise=noise_dict['rel_scint_noise'],
            )
        
        results[self.time_key] = self.bmjds[file.key]
        
        return results


    def update_unaligned_files(
        self,
        files: MEFSlice | list[MEFSlice],
        ) -> None:
        """
        Add one or more files to the list of unaligned files. Unaligned files are skipped when performing photometry.
        
        Parameters
        ----------
        files : MEFSlice | list[MEFSlice]
            The file or files.
        """
        
        if isinstance(files, MEFSlice):
            files = [files]
        
        for file in files:
            self.unaligned_files.append(file)
        
        save_unaligned_files(
            out_directory=self.out_directory,
            unaligned_files=self.unaligned_files,
        )


    def plot_noise(
        self,
        save: bool = False,
        ) -> None:
        """
        Plot the noise characterisation for each reference image.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot, by default 'False'.
        """
        
        noise_dicts = {}
        
        for key, file in self.reference_files.items():
            noise_dicts[key] = self._characterise_noise(
                file=file,
                key=key,
                )
        
        plot_noise(
            out_directory=self.out_directory,
            noise_dicts=noise_dicts,
            show=self.show_plots,
            save=save,
            )


    def _get_noise_params(
        self,
        file: MEFSlice,
        key: str,
        ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], float, float, float, float, float, NDArray[np.float64]]:
        """
        Get the noise values of a science image.
        
        Parameters
        ----------
        file : MEFSlice
            The science image file.
        key : str
            The camera:filter key.
        
        Returns
        -------
        tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], float, float, float, float, float, NDArray[np.float64]]
            The source IDs, fluxes, flux errors, number of aperture pixels, backgorund counts/pixel, bias variance, dark
            variance, flat-field variance, and scintillation noise.
        """
        
        coords = np.asarray([self.catalogs[key]['x_centroid'], self.catalogs[key]['y_centroid']]).T
        
        img, header, noise_dict = get_data(
            file=file,
            instrument=self.instrument,
            image_filter=self.image_filter,
            rebin_factor=self.rebin_factor,
            bias_corrector=self.bias_corrector,
            dark_corrector=self.dark_corrector,
            flat_corrector=self.flat_corrector,
            remove_cosmic_rays=False,
            )
        
        # global background
        bkg = self.background(img)
        n_sky = float(bkg.background_rms_median**2)  # background variance
        
        # subtract background from image
        img_clean = img - bkg.background
        
        # perform photometry
        phot = AperturePhotometer()
        phot_results = phot.compute(
            image=img_clean,
            bias_var=noise_dict['bias_var'],
            dark_var=noise_dict['dark_var'],
            flat_var=noise_dict['flat_var'],
            background_rms=np.sqrt(n_sky),
            cat_coords=coords,
            image_coords=coords,
            psf_params=self.psf_params[key],
            read_noise=self.instrument.get_read_noise(header=header),
            rel_scint_noise=noise_dict['rel_scint_noise'],
            )
        
        # get the number of pixels in the aperture
        N_pix = phot.get_aperture_area(psf_params=self.psf_params[key])
        
        fluxes = np.array(phot_results['flux'])
        flux_errs = np.array(phot_results['flux_err'])
        source_ids = np.arange(len(self.catalogs[key])) + 1
        scint_noise = noise_dict['rel_scint_noise'] * fluxes
        
        # mask unphysical flux values
        mask = fluxes > 1.
        
        return source_ids[mask], fluxes[mask], flux_errs[mask], N_pix, n_sky, float(np.median(noise_dict['bias_var'])), float(np.median(noise_dict['dark_var'])), float(np.median(noise_dict['flat_var'])), scint_noise


    def get_snrs(
        self,
        file: MEFSlice,
        key: str,
        ) -> dict[int, float]:
        """
        Get the S/N ratios for the cataloged sources in a science image.
        
        Parameters
        ----------
        file : MEFSlice
            The science image file.
        key : str
            The camera:filter key.
        
        Returns
        -------
        dict[int, float]
            The source ID (key) and S/N (value) for each source.
        """
        
        source_ids, fluxes, flux_errs, N_pix, n_sky, bias_var, dark_var, flat_var, scint_noise = self._get_noise_params(
            key=key,
            file=file,
        )
        
        snrs = np.asarray(
            snr(
                N_source=fluxes,
                N_pix=N_pix,
                n_sky=n_sky,
                bias_var=bias_var,
                dark_var=dark_var,
                flat_var=flat_var,
                read_noise=self.instrument.get_read_noise(file=file),
                scint_noise=scint_noise,
                )
            )
        
        results = {source_id: round(snr, 1) for source_id, snr in zip(source_ids, snrs)}
        
        return results


    def _characterise_noise(
        self,
        file: MEFSlice,
        key: str,
        ) -> dict[str, NDArray]:
        """
        Characterise the expected noise from an image and compare it to the measured noise for a number of cataloged 
        sources.
        
        Parameters
        ----------
        file : MEFSlice
            The science image file.
        key : str
            The camera:filter key.
        
        Returns
        -------
        dict[str, NDArray]
            The noies properties.
        """
        
        header = file.get_header()
        
        read_noise = self.instrument.get_read_noise(header=header)
        rel_scint_noise = self.instrument.get_relative_scintillation_noise(header=header)
        
        source_ids, fluxes, flux_errs, N_pix, n_sky, bias_var, dark_var, flat_var, scint_noise = self._get_noise_params(
            file=file,
            key=key,
        )
        
        N_source = np.logspace(
            np.log10(np.min(fluxes) / 1.5),
            np.log10(np.max(fluxes) * 1.5),
            100,
            )
        
        results = {}
        
        results['model_mags'] = -2.5 * np.log10(N_source)
        results['effective_noise'] = snr_stderr(
            N_source=N_source,
            N_pix=N_pix,
            n_sky=n_sky,
            bias_var=bias_var,
            dark_var=dark_var,
            flat_var=flat_var,
            read_noise=read_noise,
            rel_scint_noise=rel_scint_noise,
            )
        results['sky_noise'] = get_sky_stderr(
            N_source=N_source,
            N_pix=N_pix,
            n_sky=n_sky,
            )
        results['shot_noise'] = get_shot_stderr(N_source=N_source)
        results['bias'] = get_bias_stderr(
            N_source=N_source,
            N_pix=N_pix, bias_var=bias_var)
        results['dark_noise'] = get_dark_stderr(N_source, N_pix, dark_var)
        results['flat'] = get_flat_stderr(N_source, N_pix, flat_var)
        results['read_noise'] = get_read_stderr(N_source, N_pix, read_noise=read_noise)
        results['scint_noise'] = get_scint_stderr(N_source=N_source, rel_scint_noise=rel_scint_noise)
        
        results['measured_mags'] = -2.5 * np.log10(fluxes)
        results['measured_noise'] = counts_to_mag_factor * flux_errs / fluxes
        results['expected_measured_noise'] = snr_stderr(
            N_source=fluxes,
            N_pix=N_pix,
            n_sky=n_sky,
            bias_var=bias_var,
            dark_var=dark_var,
            flat_var=flat_var,
            read_noise=read_noise,
            rel_scint_noise=rel_scint_noise,
            )
        
        return results



################### for a clearner UI, the following functions are intentionally not Reducer methods ###################


def set_psf_params(
    aperture_selector: Callable,
    catalog: QTable,
    ) -> dict[str, float]:
    """
    Set the PSF parameters.
    
    Parameters
    ----------
    aperture_selector : Callable
        The aperture selector (e.g., `numpy.median`).
    catalog : QTable
        The source catalog.
    
    Returns
    -------
    dict[str, float]
        The PSF parameters.
    """
    
    semimajor_axis_pix = aperture_selector(catalog['semimajor_axis'].value)  # type: ignore
    semiminor_axis_pix = aperture_selector(catalog['semiminor_axis'].value)  # type: ignore
    orientation = aperture_selector(catalog['orientation'].value)  # type: ignore
    
    return {
        'semimajor_axis': semimajor_axis_pix,
        'semiminor_axis': semiminor_axis_pix,
        'orientation': orientation,
    }


def parse_alignment_results(
    results: tuple,
    camera_files: list[MEFSlice],
    transforms: dict[str, list[float]],
    unaligned_files: list[MEFSlice],
    logger: Logger,
    ) -> tuple[dict[str, list[float]], list[MEFSlice], NDArray[np.float64], dict[str, dict[str, float]]]:
    """
    Parse the alignment results.
    
    Parameters
    ----------
    results : tuple
        The alignment results.
    camera_files : list[MEFSlice]
        The files. 
    transforms : dict[Path, list[float]]
        The image-to-image alignments {file path: transform}.
    unaligned_files : list[MEFSlice]
        The files that could not be aligned.
    logger : Logger
        The logger.
    
    Returns
    -------
    tuple[dict[str, list[float]], list[str], NDArray, dict[str, float], dict[str, float]]
        The updated transforms, unaligned files, stacked image, and systematics.
    """
    
    key_transforms: dict[str, list[float]] = {}
    key_unaligned_files: list[MEFSlice] = []
    key_systematics: dict[str, dict[str, float]] = {}
    queued_logs: list[tuple[str, str]] = []
    
    # unpack results
    batch_stacked_images, batch_transforms, batch_systematics, batch_queued_logs = zip(*results)
    
    # combine results
    for i in range(len(batch_stacked_images)):
        key_transforms.update(batch_transforms[i])
        key_systematics.update(batch_systematics[i])
        queued_logs += batch_queued_logs[i]
    
    # write log messages
    write_queued_logs(
        queued_logs=queued_logs,
        logger=logger,
        )
    
    aligned_files = list(key_transforms.keys())
    for file in camera_files:
        if file.key not in aligned_files:
            key_unaligned_files.append(file)
    
    stacked_image = np.sum(batch_stacked_images, axis=0)  # stack images
    
    transforms.update(key_transforms)  # update transforms to include current filter
    unaligned_files += key_unaligned_files  # update unaligned files
    
    logger.info(f"Done.")
    logger.info(f'{len(key_transforms)} image(s) aligned.')
    logger.info(f'{len(key_unaligned_files)} image(s) could not be aligned.')
    
    return transforms, unaligned_files, stacked_image, key_systematics


def write_queued_logs(
    queued_logs: list[tuple[str, str]],
    logger: Logger,
    ) -> None:
    """
    Write queued logs to file.
    
    Parameters
    ----------
    queued_logs : list[tuple[str, str]]
        The queued logs (level, log).
    logger : Logger
        The logger.
    
    Raises
    ------
    ValueError
        If the log level is not recognised.
    """
    
    for queued_log in queued_logs:
        if len(queued_log) > 0:
            level, message = queued_log
            if level.lower() == 'debug':
                logger.debug(message)
            elif level.lower() == 'info':
                logger.info(message)
            elif level.lower() == 'warning':
                logger.warning(message)
            elif level.lower() == 'error':
                logger.error(message)
            elif level.lower() == 'critical':
                logger.critical(message)
            else:
                raise ValueError(f'[PHOPTIC] Unrecognised log level {level}.')


def save_unaligned_files(
    out_directory: Path,
    unaligned_files: list[MEFSlice],
    ) -> None:
    """
    Save the unaligned files to a text file.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    unaligned_files : list[MEFSlice]
        The list of unaligned files.
    """
    
    if len(unaligned_files) > 0:
        with open(os.path.join(out_directory, "diag/unaligned_files.txt"), "w") as unaligned_file:
            for file in unaligned_files:
                unaligned_file.write(str(file.key) + "\n")


def create_targets_dict(
    catalogs: dict[str, QTable],
    ) -> dict[str, list[int]]:
    """
    Create a dictionary of target IDs for all catalog sources.
    
    Parameters
    ----------
    catalogs : dict[str, QTable]
        The catalogs.
    
    Returns
    -------
    dict[str, list[int]]
        The target IDs for all catalog sources.
    """
    
    targets: dict[str, list[int]] = {}
    
    for key, cat in catalogs.items():
        targets[key] = []
        for i in range(len(cat)):
            targets[key].append(i + 1)
    
    return targets


def save_photometry_results(
    results: tuple[dict],
    catalogs: dict[str, QTable],
    barycenter: bool,
    save_dir: Path,
    key: str,
    ):
    """
    Save the photometry results to disk.
    
    Parameters
    ----------
    results : tuple[dict]
        The photometry results.
    catalogs : dict[str, QTable]
        The source catalogs.
    save_dir : Path
        The save directory path.
    key : str
        The camera:filter key.
    """
    
    photometry_results = parse_photometry_results(results)
    
    time_key = 'BMJD' if barycenter else 'MJD'
    
    # for each source in the catalog
    for i in range(len(catalogs[key])):
        
        # unpack results for ith source
        source_results = {}
        for phot_key, phot_values in photometry_results.items():
            
            # time is a special case since it is already a single column
            if phot_key == time_key:
                source_results[phot_key] = np.asarray(phot_values)
            # for other keys, the ith column needs to be extracted
            else:
                col = [phot_value[i] for phot_value in phot_values]
                source_results[phot_key] = np.asarray(col)
        
        # define light curve as a DataFrame
        df = pd.DataFrame(source_results)
        
        # drop NaNs
        df.dropna(inplace=True, ignore_index=True)
        
        # make time column left-most column
        time_col = df.pop(time_key)
        df.insert(0, time_key, time_col)
        
        # save to file
        df.to_csv(
            os.path.join(
                save_dir,
                f'{key}_source_{i + 1}.csv',
                ),
            index=False,
            )


def parse_photometry_results(
    results: tuple[dict[str, list]],
    ) -> dict[str, list[list[float]]]:
    """
    Merge the multiprocessed photometry results into a single dictionary.
    
    Parameters
    ----------
    results : tuple[dict[str, list]]
        The multiprocessed photometry results.
    
    Returns
    -------
    dict[str, list[list[float]]]
        The photometry results in a single dictionary.
    """
    
    photometry_results = {}
    for result in results:
        for key, value in result.items():
            if key not in photometry_results:
                photometry_results[key] = []
            photometry_results[key].append(value)
    
    return photometry_results


def save_catalog(
    catalog: QTable,
    key: str,
    out_directory: Path,
    ) -> None:
    """
    Save a camera:filter catalog to file using `astropy`'s ECSV format.
    
    Parameters
    ----------
    catalog : QTable
        The catalog for a specific camera:filter combination.
    key : str
        The camera:filter key.
    out_directory : Path
        The output directory.
    """
    
    catalog.write(
        out_directory / 'cat' / f'{key}_catalog.ecsv',
        format='ascii.ecsv',
        overwrite=True,
        )