import os.path
from pathlib import Path
from typing import Any, Literal


from astropy import units as u
from astropy.table import QTable
from astropy.timeseries import TimeSeries
from astropy.time import Time
from astropy.visualization import simple_norm, PercentileInterval
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from photutils.aperture import ApertureStats, BoundingBox


from phoptic.background.global_background import BaseBackground
from phoptic.instruments import Instrument
from phoptic.photometers import AperturePhotometer, get_growth_curve
from phoptic.fitting.models import gaussian
from phoptic.fitting.routines import fit_rms_vs_flux
from phoptic.utils.constants import catalog_colors, fwhm_scale
from phoptic.utils.helpers import save_figure
from phoptic.mef_slice import MEFSlice
from phoptic.timing.timeseries import get_lc
from phoptic.utils.helpers import camera_key, sort_dict_by_filters




def plot_catalogs(
    out_directory: Path,
    stacked_images: dict[str, NDArray],
    catalogs: dict[str, QTable],
    show: bool,
    save: bool,
    percentile: float | None = None,
    return_fig: bool = False,
    ) -> Figure | None:
    """
    Plot the source catalogs.
    
    Parameters
    ----------
    out_directory : Path
        The path to the directory in which the resulting plot will be saved.
    stacked_images : dict[str, NDArray]
        The stacked images for each filter {filter: image}.
    catalogs : dict[str, QTable]
        The source catalogs for each filter {filter: catalog}.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    return_fig : bool, optional
        Whether to return the figure, by default `False`.
    
    Returns
    -------
    Figure | None
        If `return_fig=True`, the resulting figure is returned. Otherwise, nothing is returned.
    """
    
    ncols: int = len(stacked_images)
    
    fig, axes = plt.subplots(
        ncols=ncols,
        tight_layout=True,
        figsize=(ncols * 5, 5),
        )
    
    if ncols == 1:
        axes = [axes]
    
    for i, fltr in enumerate(stacked_images):
        
        plot_image = np.clip(stacked_images[fltr], 0, None)  # clip negative values to zero for better visualisation
        
        # plot stacked image
        if percentile is None:
            axes[i].imshow(
                plot_image,
                origin="lower",
                cmap="Greys",
                interpolation="nearest",
                norm=simple_norm(
                    plot_image,
                    stretch="log",
                    ),
                )
        else:
            interval = PercentileInterval(percentile)
            vmin, vmax = interval.get_limits(stacked_images[fltr])
            axes[i].imshow(
                plot_image,
                origin="lower",
                cmap="Greys",
                vmin=vmin,
                vmax=vmax,
                )
        
        # get aperture radius
        radius = 5 * np.median(catalogs[fltr]["semimajor_axis"].value)  # type: ignore
        
        for j in range(len(catalogs[fltr])):
            # label sources
            axes[i].add_patch(
                Circle(
                    xy=(
                        catalogs[fltr]["x_centroid"][j],
                        catalogs[fltr]["y_centroid"][j],
                        ),  # type: ignore
                    radius=radius,
                    edgecolor=catalog_colors[j % len(catalog_colors)],
                    facecolor="none",
                    lw=1,
                    ),
                )
            axes[i].text(
                catalogs[fltr]["x_centroid"][j] + 1.05 * radius,
                catalogs[fltr]["y_centroid"][j] + 1.05 * radius,
                j + 1,  # source number
                color=catalog_colors[j % len(catalog_colors)],
                fontsize='large',
                )
            
            # label plot
            axes[i].set_title(fltr, fontsize='large')
            axes[i].set_xlabel("X", fontsize='large')
            axes[i].set_ylabel("Y", fontsize='large')
    
    if save:
        save_figure(
            fig=fig,
            path=out_directory / 'cat' / 'catalogs.pdf',
        )
    
    if show:
        plt.show(fig)
    
    if return_fig:
        return fig


def plot_time_between_files(
    out_directory: Path,
    camera_files: dict[str, list[MEFSlice]],
    bmjds: dict[str, float],
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the times between files. Useful for identifying gaps.
    
    Parameters
    ----------
    out_directory : Path
        The directory path to which the resulting plot will be saved.
    camera_files : dict[str, list[MEFSlice]]
        The files separated by camera.
    bmjds : dict[str, float]
        The file time stamps {file path + extension: time stamp}.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    ncols: int = len(camera_files)
    
    fig, axes = plt.subplots(
        nrows=3,
        ncols=ncols,
        tight_layout=True,
        figsize=((2 * ncols / 3) * 6.4, 2 * 4.8),
        sharey='row',
        gridspec_kw={
            'wspace': 0,
            },
        )
    
    for fltr in list(camera_files.keys()):
        times = np.array([bmjds[file.key] for file in camera_files[fltr]])
        times -= times.min()
        times *= 86400  # convert to seconds from first observation
        dt = np.diff(times)  # get time between files
        file_numbers = np.arange(2, len(times) + 1, 1)  # start from 2 because we are plotting the time between files
        
        bin_edges = np.arange(int(dt.min()), np.ceil(dt.max() + .2), .1)  # define bins with width 0.1 s
        
        if len(camera_files) == 1:
            axes[0].set_title(fltr)
            
            # cumulative plot of time between files
            axes[0].plot(file_numbers, np.cumsum(dt), "k-", lw=1)
            
            # time between each file
            axes[1].plot(file_numbers, dt, "k-", lw=1)
            
            axes[2].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
            axes[2].set_yscale("log")
            
            axes[0].set_ylabel("Cumulative time between files [s]")
            axes[0].set_xlabel("File number")
            
            axes[1].set_ylabel("Time between files [s]")
            axes[1].set_xlabel("File number")
            
            axes[2].set_xlabel("Time between files [s]")
        else:
            axes[0, list(camera_files.keys()).index(fltr)].set_title(fltr)
            
            # cumulative plot of time between files
            axes[0, list(camera_files.keys()).index(fltr)].plot(file_numbers, np.cumsum(dt), "k-", lw=1)
            
            # time between each file
            axes[1, list(camera_files.keys()).index(fltr)].plot(file_numbers, dt, "k-", lw=1)
            
            # histogram of time between files
            axes[2, list(camera_files.keys()).index(fltr)].hist(dt, bins=bin_edges, histtype="step", color="black", lw=1)
            axes[2, list(camera_files.keys()).index(fltr)].set_yscale("log")
            
            axes[0, 0].set_ylabel("Cumulative time between files [s]")
            axes[1, 0].set_ylabel("Time between files [s]")
            
            for col in range(len(camera_files)):
                axes[0, col].set_xlabel("File number")
                axes[1, col].set_xlabel("File number")
                axes[2, col].set_xlabel("Time between files [s]")
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    
    if save:
        save_figure(
            fig=fig,
            path=out_directory / 'diag' / 'header_times.pdf',
            )
    
    if show:
        plt.show(fig)
    else:
        plt.close(fig)


def plot_systematics(
    out_directory: Path,
    instrument: Instrument,
    bin_factor: int,
    t_ref: float,
    show: bool,
    save: bool,
    time_key: Literal['BMJD', 'MJD'] = 'BMJD',
    ) -> None:
    """
    Plot the time-varying systematics for each camera.
    
    Parameters
    ----------
    out_directory : Path
        The directory to which the background files, and where the resulting plot will be saved if `save=True`.
    instrument : Instrument
        The instrument used to make the observation.
    bin_factor : int
        The effective binning factor of the image. This is the product of hardware pixel binning factor and the 
        software pixel binning factor.
    t_ref : float
        The reference time.
    time_key : Literal['BMJD', 'MJD'], optional
        The time key, by default "BMJD".
    show: bool
        Whether to display the plot.
    save : bool
        Whether to save the plot.
    """
    
    diag_files = os.listdir(os.path.join(out_directory, 'diag'))
    
    systematics_files = {}
    for file in diag_files:
        if file.endswith('_systematics.csv'):
            key = file.split('_')[0]
            systematics_files[key] = os.path.join(out_directory, f'diag/{file}')
    systematics_files = sort_dict_by_filters(systematics_files)
    
    fig, axes = plt.subplots(
        nrows=4,
        ncols=len(systematics_files),
        tight_layout=True,
        figsize=( 2 / 3 * len(systematics_files) * 6.4, 1.6 * 4.8),
        sharex='col',
        gridspec_kw={
            'hspace': 0,
            },
        )
    
    for col, (key, file) in enumerate(systematics_files.items()):
        df = pd.read_csv(file)
        
        t = np.asarray(df[time_key].values)
        plot_times = (t - t_ref) * 86400
        
        if len(systematics_files) == 1:
            axes[0].set_title(key, fontsize='large')
            axes[0].plot(plot_times, df['bkg_median'].values, "k.", ms=2)
            axes[1].plot(plot_times, df['bkg_rms'].values, "k.", ms=2)
            axes[2].plot(plot_times, df['FWHM'].values * bin_factor * instrument.pixel_scales[camera_key(key)], "k.", ms=2)
            axes[3].plot(plot_times, df['airmass'].values, "k.", ms=2)
            # axes[4].plot(plot_times, 100 * df['rel_scint_noise'].values, "k.", ms=2)
            
            # label plots
            axes[-1].set_xlabel(f"Time from {time_key} {t_ref:.4f} [s]", fontsize='large')
            axes[0].set_ylabel("BKG [e$^-$/pix]", fontsize='large')
            axes[1].set_ylabel("$\\sigma_{\\rm BKG}$ [e$^-$/pix]", fontsize='large')
            axes[2].set_ylabel("FWHM ['']", fontsize='large')
            axes[3].set_ylabel("Airmass", fontsize='large')
            # axes[4].set_ylabel("$\\sigma_{\\rm scint}$ [%]", fontsize='large')
        else:
            axes[0, col].set_title(key, fontsize='large')
            axes[0, col].plot(plot_times, df['bkg_median'].values, "k.", ms=2)
            axes[1, col].plot(plot_times, df['bkg_rms'].values, "k.", ms=2)
            axes[2, col].plot(plot_times, df['FWHM'].values * bin_factor * instrument.pixel_scales[camera_key(key)], "k.", ms=2)
            axes[3, col].plot(plot_times, df['airmass'].values, "k.", ms=2)
            # axes[4, col].plot(plot_times, 100 * df['rel_scint_noise'].values, "k.", ms=2)
            
            # label plots
            axes[-1, col].set_xlabel(f"Time from {time_key} {t_ref:.4f} [s]", fontsize='large')
            axes[0, col].set_ylabel("BKG [e$^-$/pix]", fontsize='large')
            axes[1, col].set_ylabel("$\\sigma_{\\rm BKG}$ [e$^-$/pix]", fontsize='large')
            axes[2, col].set_ylabel("FWHM ['']", fontsize='large')
            axes[3, col].set_ylabel("Airmass", fontsize='large')
            # axes[4, col].set_ylabel("$\\sigma_{\\rm scint}$ [%]", fontsize='large')
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)
    
    if save:
        save_figure(
            fig=fig,
            path=out_directory / 'diag' / 'systematics.pdf',
            )
    
    if show:
        plt.show()
    else:
        fig.clear()
        plt.close(fig)


def plot_background_meshes(
    out_directory: Path,
    images: dict[str, NDArray[np.float64]],
    background: BaseBackground,
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the background mesh on top a series of images.
    
    Parameters
    ----------
    out_directory : Path
        The path to the output directory.
    images : dict[str, NDArray[np.float64]]
        The images {string: image}
    background: BaseBackground
        The background estimator.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    ncols = len(images)
    fig, axes = plt.subplots(ncols=ncols, tight_layout=True, figsize=(ncols * 5, 5))
    
    if ncols == 1:
        # convert axes to list
        axes = [axes]
    
    for i, (label, image) in enumerate(images.items()):
        
        # clip negative values
        plot_image = np.clip(image, 0., None)
        
        bkg = background(image)
        
        # plot background mesh
        axes[i].imshow(
            plot_image,
            origin="lower",
            cmap="Greys",
            interpolation="nearest",
            norm=simple_norm(plot_image, stretch="log"),
            )
        bkg.plot_meshes(
            ax=axes[i],
            outlines=True,
            marker='.',
            color='red',
            alpha=0.3,
            )
        
        #label plot
        axes[i].set_title(label)
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
    
    if save:
        save_figure(
            fig=fig,
            path=out_directory / 'diag' / 'background_meshes.pdf',
            )
    
    if show:
        plt.show(fig)
    else:
        fig.clear()
        plt.close(fig)


def plot_growth_curves(
    image: NDArray,
    cat: QTable,
    targets: int | list[int],
    psf_params: dict,
    ) -> Figure:
    """
    Plot the growth curves given a (stacked) image and corresponding source catalog.
    
    Parameters
    ----------
    image : NDArray
        The image.
    cat : QTable
        The catalog corresponding to `image`.
    targets : int | list[int]
        The target(s) for which growth curves are to be computed.
    psf_params : dict
        The PSF parameters.
    
    Returns
    -------
    Figure
        The growth curve plots.
    """
    
    def pix2sigma(x):
        return x / (psf_params['semimajor_axis'] * fwhm_scale)
    
    def sigma2pix(x):
        return x * (psf_params['semimajor_axis'] / fwhm_scale)
    
    if isinstance(targets, int):
        targets = [targets]
    
    n = len(targets)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(cols * 3, rows * 3),
        tight_layout=True,
        sharey='row',
    )
    
    if rows > 1 and cols > 1:
        for row in axes:
            row[0].set_ylabel('Flux [%]', fontsize='large')
    elif cols > 1:
        for col in axes:
            col.set_ylabel('Flux [%]', fontsize='large')
    else:
        axes.set_ylabel('Flux [%]', fontsize='large')
    
    axes = np.asarray([axes]).flatten()
    
    for target in targets:
        i = targets.index(target)
        
        radii, fluxes = get_growth_curve(
            image=image,
            x_centroid=cat['x_centroid'][i],
            y_centroid=cat['y_centroid'][i],
            r_max = round(10 * psf_params['semimajor_axis']),
        )
        
        axes[i].step(
            radii,
            100 * fluxes / np.max(fluxes),
            c='k',
            lw=1,
            where='mid',
            )
        
        secax = axes[i].secondary_xaxis('top', functions=(pix2sigma, sigma2pix))
        secax.set_xlabel('Radius [FWHM]', fontsize='large')
        secax.minorticks_on()
        secax.tick_params(which='both', direction='in')
        
        axes[i].set_title(f'Source {target}', fontsize='large')
        axes[i].set_xlabel('Radius [pixels]', fontsize='large')
        
        axes[i].minorticks_on()
        axes[i].tick_params(which='both', direction='in', right=True)
    
    # delete empty subplots
    m = axes.size - n
    for i in range(1, m + 1):
        fig.delaxes(axes[-i])
    
    return fig


def plot_psf(
    catalog: QTable,
    source_indx: int,
    stacked_image: NDArray,
    key: str,
    a: float,
    b: float,
    out_directory: Path,
    ) -> None:
    """
    Plot the PSF for given source.
    
    Parameters
    ----------
    catalog : QTable
        The source catalog.
    source_indx : int
        The index of the source in the catalog.
    stacked_image : NDArray
        The catalog image.
    key : str
        The camera:filter key.
    a : float
        The semimajor standard deviation of the PSF.
    b : float
        The semiminor standard deviation of the PSF.
    out_directory : Path,
        The save path.
    """
    
    x_lo, x_hi = 0, stacked_image.shape[1]
    y_lo, y_hi = 0, stacked_image.shape[0]
    
    w = a * 10  # region width
    
    xc = catalog['x_centroid'][source_indx]
    yc = catalog['y_centroid'][source_indx]
    x_range = np.arange(max(x_lo, round(xc - w)), min(x_hi, round(xc + w)))  # x range
    y_range = np.arange(max(y_lo, round(yc - w)), min(y_hi, round(yc + w)))  # y range
    x_smooth = np.linspace(x_range[0], x_range[-1], 100)
    y_smooth = np.linspace(y_range[0], y_range[-1], 100)
    
    theta = catalog['orientation'].value[source_indx]
    theta_rad = theta * np.pi / 180
    
    # create mask
    mask = np.zeros_like(stacked_image, dtype=bool)
    for x_ in x_range:
        for y_ in y_range:
            mask[y_, x_] = True
    
    # isolate source
    rows_to_keep = np.any(mask, axis=1)
    region = stacked_image[rows_to_keep, :]
    cols_to_keep = np.any(mask, axis=0)
    region = region[:, cols_to_keep]
    
    fig, axes = plt.subplots(
        ncols=2,
        nrows=2,
        tight_layout=True,
        figsize=(6, 6),
        sharex='col',
        sharey='row',
        gridspec_kw={
            'hspace': 0,
            'wspace': 0,
            },
        )
    fig.delaxes(axes[0, 1])
    
    x, y = np.meshgrid(x_range, y_range)
    axes[1, 0].contour(
        x,
        y,
        region,
        5,
        colors='black',
        linewidths=1,
        zorder=1,
        linestyles='dashdot',
        )
    axes[1, 0].set_xlabel('X', fontsize='large')
    axes[1, 0].set_ylabel('Y', fontsize='large')
    axes[1, 0].add_patch(
        Ellipse(
            xy=(xc, yc),
            width=2 * fwhm_scale * a,  # in this parameterisation, the width is the semimajor axis
            height=2 * fwhm_scale * b,  # in this parameterisation, the height is the semiminor axis
            angle=theta,  # in this parameterisation, the angle is the orientation of the PSF
            facecolor='none',
            edgecolor='r',
            lw=1,
            ls='-',
            zorder=2,
            ),
        )
    
    # project PSF onto x, y axes
    xstd = np.sqrt(a**2 * np.cos(theta_rad)**2 + b**2 * np.sin(theta_rad)**2)
    ystd = np.sqrt(a**2 * np.sin(theta_rad)**2 + b**2 * np.cos(theta_rad)**2)
    
    axes[0, 0].step(
        x_range,
        100 * region[region.shape[0] // 2, :] / np.max(region[region.shape[0] // 2, :]),
        color='k',
        lw=1,
        where='mid',
        zorder=1,
        )
    axes[0, 0].plot(
        x_smooth,
        gaussian(x_smooth, 100, xc, xstd),
        'r-',
        lw=1,
        zorder=2,
    )
    axes[0, 0].set_ylabel('Peak flux [%]', fontsize='large')
    
    axes[1, 1].step(
        100 * region[:, region.shape[1] // 2] / np.max(region[:, region.shape[1] // 2]),
        y_range,
        color='k',
        lw=1,
        where='mid',
        )
    axes[1, 1].plot(
        gaussian(y_smooth, 100, yc, ystd),
        y_smooth,
        'r-',
        lw=1,
    )
    axes[1, 1].set_xlabel('Peak flux [%]', fontsize='large')
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(
            which='both',
            direction='in',
            right=True,
            top=True,
            )
    
    fig.suptitle(f'{key} Source {source_indx + 1}', fontsize='large')
    save_figure(
        fig=fig,
        path=out_directory / 'psfs' / f'{key}_source_{source_indx + 1}.pdf',
        )
    plt.close(fig)


def plot_rms_vs_median_flux(
    lc_dir: Path,
    save_dir: Path,
    phot_label: str,
    show: bool = True,
    ) -> None:
    """
    Plot the RMS as a function of the median flux for all catalog sources.
    
    Parameters
    ----------
    lc_dir : Path
        The light curve directory path.
    save_dir : Path
        The output directory path.
    phot_label : str
        The photometry label.
    show : bool, optional
        Whether to show the plot, by default True.
    """
    
    data: dict[str, dict[str, dict[str, float]]] = get_lc_rms_and_flux_dict(lc_dir=lc_dir)
    pl_fits: dict[str, dict[str, NDArray[np.float64]]] = fit_rms_vs_flux(data)
    
    ncols: int = len(pl_fits)
    assert ncols > 0, f"[PHOPTIC] No valid light curve files found in {lc_dir}."
    
    fig, axes = plt.subplots(
        nrows=2,
        ncols=ncols,
        tight_layout=True,
        figsize=(2 / 3 * ncols * 6.4, 4.8),
        sharex='col',
        sharey='row',
        squeeze=False,
        gridspec_kw={
            'hspace': 0,
            'wspace': 0,
            'height_ratios': [4, 1],
            },
        )
    
    for i, key in enumerate(data.keys()):
        if i == 0:
            axes[0][i].set_ylabel(
                'Flux RMS [counts]',
                fontsize='large',
                )
            axes[1][i].set_ylabel(
                '$\\frac{\\rm RMS}{\\rm model}$',
                fontsize='xx-large',
                )
        
        axes[1][i].set_xlabel(
            'Median flux [counts]',
            fontsize='large',
            )
        
        axes[0][i].set_title(
            key,
            fontsize='large',
            )
        
        # plot model
        axes[0][i].plot(
            pl_fits[key]['flux'],
            pl_fits[key]['rms'],
            color='blue',
            lw=1,
            )
        axes[0][i].fill_between(
            pl_fits[key]['flux'],
            pl_fits[key]['rms'] - pl_fits[key]['err'],
            pl_fits[key]['rms'] + pl_fits[key]['err'],
            color='grey',
            edgecolor='none',
            alpha=.5,
            )
        
        ratios = []
        # highlight potentially variable sources
        for source_number, values in data[key].items():
            # get index of current source
            j = np.where(pl_fits[key]['ids'] == int(source_number))[0]
            
            r = values['rms'] / pl_fits[key]['rms'][j]
            ratios.append(r)
            
            if r - 1 >= pl_fits[key]['err'][j] / pl_fits[key]['rms'][j]:
                color = 'red'
            else:
                color = 'black'
            
            axes[0][i].scatter(
                values['flux'],
                values['rms'],
                marker='.',
                color=color,
                )
            axes[0][i].text(
                values['flux'] * 1.03,
                values['rms'] * 1.03,
                str(source_number),
                color=color,
                fontsize='large',
                )
            
            axes[1][i].scatter(
                values['flux'],
                r,
                marker='.',
                color=color,
                )
            axes[1][i].text(
                values['flux'] * 1.015,
                r * 1.015,
                str(source_number),
                fontsize='large',
                color=color,
                )
        
        axes[0][i].set_yscale('log')
        
        axes[1][i].plot(
            pl_fits[key]['flux'],
            np.ones_like(pl_fits[key]['flux']),
            color='blue',
            lw=1,
            )
        axes[1][i].fill_between(
            pl_fits[key]['flux'],
            1 - pl_fits[key]['err'] / pl_fits[key]['rms'],
            1 + pl_fits[key]['err'] / pl_fits[key]['rms'],
            color='grey',
            edgecolor='none',
            alpha=.5,
            )
        
        lo = np.min(ratios) * .75
        hi = np.max(ratios) * 1.25
        ax_lo, ax_hi = axes[1][i].get_ylim()
        if lo < ax_lo and hi > ax_hi:
            axes[1][i].set_ylim(lo, hi)
        elif hi > ax_hi:
            axes[1][i].set_ylim(ax_lo, hi)
        elif lo < ax_lo:
            axes[1][i].set_ylim(lo, ax_hi)
    
    for ax in axes.flatten():
        ax.set_xscale('log')
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
    
    save_figure(
        fig=fig,
        path=save_dir / f'{phot_label}_rms_vs_median.pdf',
        )
    
    if show:
        plt.show(fig)
    else:
        plt.close(fig)


def get_lc_rms_and_flux_dict(
    lc_dir: Path,
    ) -> dict[str, dict[str, dict[str, float]]]:
    """
    Get the RMS and median flux for a series of light curves.
    
    Parameters
    ----------
    lc_dir : Path
        The directory path to the light curves.
    
    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        The median and RMS flux values for each light curve grouped by filter.
    """
    
    lcs = os.listdir(lc_dir)
    
    data = {}
    
    for lc in lcs:
        
        file_name, extension = lc.split('.')
        fltr, _, source_number = file_name.split('_')
        
        df = pd.read_csv(os.path.join(lc_dir, lc))
        
        flux = np.array(df['flux'].values, dtype=np.float64)
        flux = flux
        
        median = np.median(flux)
        if not np.isfinite(np.log10(median)):
            continue
        
        rms = np.std(flux)
        if not np.isfinite(np.log10(rms)):
            continue
        
        if fltr not in data.keys():
            data[fltr] = {}
        source_info = {
            'rms': rms,
            'flux': median,
            }
        data[fltr][source_number] = source_info
    
    return sort_dict_by_filters(data)


def plot_snrs(
    out_directory: Path,
    snrs: dict[str, dict[int, float]],
    show: bool,
    save: bool,
    ) -> None:
    """
    Plot the S/N for each source.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    snrs : dict[str, dict[int, float]]
        The S/N for each source in each catalog.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    ncols: int = len(snrs)
    
    fig, axes = plt.subplots(
        ncols=ncols,
        tight_layout=True,
        figsize=(2 / 3 * ncols * 6.4, 5),
        )
    
    # in event of a single column, make axes subscriptable
    if ncols == 1:
        axes = [axes]
    
    for i, (key, snr_dict) in enumerate(snrs.items()):
        
        axes[i].set_title(
            key,
            fontsize='large',
            )
        axes[i].set_xlabel(
            'Source ID',
            fontsize='large',
            )
        axes[i].set_ylabel(
            'S/N',
            fontsize='large',
            )
        
        p = axes[i].bar(
            snr_dict.keys(),
            snr_dict.values(),
            facecolor='none',
            edgecolor='k',
            lw=1,
            )
        axes[i].bar_label(
            p,
            padding=0.02 * axes[i].get_ylim()[1],
            fontsize='large',
            rotation=90,
            )
    
    for ax in axes:
        ax.set_ylim(ax.get_ylim()[0], 1.2 * ax.get_ylim()[1])
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', right=True, top=True)
    
    if save:
        save_figure(
            fig=fig,
            path=out_directory / 'diag' / 'snrs.pdf',
            )
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_noise(
    out_directory: Path,
    noise_dicts: dict[str, dict[str, NDArray[np.float64]]],
    show: bool,
    save: bool,
    ):
    """
    Plot the various noise contributions and compare them to the measured noise for a series of images.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    noise_dicts : dict[str, dict[str, NDArray[np.float64]]]
        The noise dictionaries for each camera.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot.
    """
    
    ncols: int = len(noise_dicts)
    
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=2,
        squeeze=False,
        tight_layout=True,
        sharex='col',
        sharey='row',
        gridspec_kw={
            'hspace': 0,
            'wspace': 0,
            'height_ratios': [4, 1],
            },
        figsize=(2 / 3 * ncols * 6.4, 4.8),
        )
    
    for i, (key, results) in enumerate(noise_dicts.items()):
        
        axes[0][i].plot(results['model_mags'], results['effective_noise'], label='Effective noise', c='k', lw=1, zorder=3)
        
        axes[0][i].plot(results['model_mags'], results['sky_noise'], ls=(5, (10, 3)), lw=1, label='Sky noise')
        axes[0][i].plot(results['model_mags'], results['shot_noise'], ls=(0, (5, 5)), lw=1, label='Shot noise')
        
        if np.any(results['bias'] > 0):
            axes[0][i].plot(results['model_mags'], results['bias'], ls=(0, (5, 1)), lw=1, label='Bias')
        
        if np.any(results['dark_noise'] > 0):
            axes[0][i].plot(results['model_mags'], results['dark_noise'], ls=(0, (3, 5, 1, 5)), lw=1, label='Dark noise')
        
        if np.any(results['flat'] > 0):
            axes[0][i].plot(results['model_mags'], results['flat'], ls=(0, (3, 1, 1, 1)), lw=1, label='Flat')
        
        axes[0][i].plot(results['model_mags'], results['read_noise'], ls=(0, (3, 5, 1, 5, 1, 5)), lw=1, label='Read noise')
        axes[0][i].plot(results['model_mags'], results['scint_noise'], ls=(0, (1, 10)), lw=1, label='Scintillation')
        
        axes[0][i].scatter(
            results['measured_mags'],
            results['measured_noise'],
            label='Measured'
            )
        
        axes[1][i].axhline(
            1,
            c='k',
            lw=1,
            )
        axes[1][i].scatter(
            results['measured_mags'],
            results['measured_noise'] / results['expected_measured_noise'],
            )
        axes[1][i].fill_between(
            axes[1][i].set_xlim(),
            [1.05, 1.05],
            [.95, .95],
            color='grey',
            edgecolor='none',
            alpha=.5,
            )
        
        for j in range(len(results['measured_mags'])):
            axes[0][i].text(
                results['measured_mags'][j],
                results['measured_noise'][j] * 1.2,
                f'{j + 1}',
                ha='center',
                va='bottom',
                fontsize='large',
                )
            
            r = results['measured_noise'][j] / results['expected_measured_noise'][j]
            
            if r >= 1:
                axes[1][i].text(
                results['measured_mags'][j],
                r * 1.01,
                f'{j + 1}',
                ha='center',
                va='bottom',
                fontsize='large',
                )
            else:
                axes[1][i].text(
                results['measured_mags'][j],
                r * .99,
                f'{j + 1}',
                ha='center',
                va='top',
                fontsize='large',
                )
        
        axes[0][i].set_yscale('log')
        axes[0][i].set_title(key, fontsize='large')
        
        axes[1][i].set_xlabel('-2.5 log(counts)', fontsize='large')
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', right=True, top=True)
    
    for ax in axes[0, :]:
        ax.invert_xaxis()
    
    axes[0, 0].set_ylabel('$\\sigma_{\\rm mag}$', fontsize='large')
    axes[1, 0].set_ylabel('$\\frac{\\sigma_{\\rm measured}}{\\sigma_{\\rm expected}}$', fontsize='xx-large')
    
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        bbox_to_anchor=(.5, .97),
        loc='lower center',
        ncol=len(results),
        bbox_transform=fig.transFigure,
        fontsize='large',
        )
    
    if save:
        save_figure(
            fig=fig,
            path=out_directory / 'diag' / 'noise_characterisation.pdf',
            )
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_apertures(
    out_directory: Path,
    data: NDArray,
    cat: QTable,
    targets: list[int] | int,
    photometer: AperturePhotometer,
    psf_params: dict[str, float],
    key: str,
    show: bool,
    save: bool,
    ):
    """
    Plot the specified aperture over each target source.
    
    Parameters
    ----------
    out_directory : Path
        The output directory. Used to save the plot if `save=True`.
    data : NDArray
        The image data.
    cat : QTable
        The source catalog.
    targets : list[int] | int
        The target IDs to plot apertures for.
    photometer : AperturePhotometer
        The `AperturePhotometer` instance.
    psf_params : dict[str, float]
        The PSF parameters.
    key : str
        The camera:filter key.
    show : bool
        Whether to show the plot.
    save : bool
        Whether to save the plot. If true, the plot is saved to `out_directory/diag/apertures/fltr_apertures.pdf`.
    """
    
    if isinstance(targets, int):
        targets = [targets]
    
    n = len(targets)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3, nrows * 3),
        tight_layout=True,
    )
    
    axes = np.asarray([axes]).flatten()
    
    # delete axes that will not be used
    excess_axes = axes.size - n
    for i in range(1, 1 + excess_axes):
        fig.delaxes(axes[-i])
    
    region_size = get_max_region_size(
        targets=targets,
        photometer=photometer,
        data=data,
        cat=cat,
        psf_params=psf_params,
    )
    
    for i, target in enumerate(targets):
        cat_indx = target - 1
        position = [cat['x_centroid'][cat_indx], cat['y_centroid'][cat_indx]]
        theta = cat['orientation'][cat_indx].value
        
        aperture = photometer.get_aperture(
            position=position,
            psf_params=psf_params,
            )
        
        if photometer.local_background_estimator is not None:
            annulus_stats = photometer.local_background_estimator.get_stats(
                data=data,
                position=position,
                semimajor_axis=psf_params['semimajor_axis'],
                semiminor_axis=psf_params['semiminor_axis'],
                theta=theta * np.pi / 180,  # radians
                )
            bbox = annulus_stats.bbox
            
            padding = region_size // 10
        else:
            aperture_stats = ApertureStats(
                data=data,
                aperture=aperture,
            )
            bbox = aperture_stats.bbox
            
            padding = region_size // 4
        
        ixmin = max(0, bbox.ixmin - padding)
        ixmax = min(data.shape[1], bbox.ixmin + region_size + padding)
        dx = bbox.ixmin - ixmin
        
        iymin = max(0, bbox.iymin - padding)
        iymax = min(data.shape[0], bbox.iymin + region_size + padding)
        dy = bbox.iymin - iymin
        
        bbox = BoundingBox(
            ixmin=ixmin,
            ixmax=ixmax,
            iymin=iymin,
            iymax=iymax,
            )
        
        # get region of interest
        region = data[bbox.iymin:bbox.iymax, bbox.ixmin:bbox.ixmax]
        centre = (position[0] - bbox.ixmin, position[1] - bbox.iymin)  # centre of region
        
        axes[i].imshow(region, origin='lower', cmap='Greys', norm=simple_norm(region, stretch='log'))
        
        if photometer.local_background_estimator is not None:
            
            annulus_mask = np.asarray(annulus_stats.data_cutout).astype(bool)
            
            # factor of 2 since matplotlib assumes diameter
            annulus_inner_width = 2 * photometer.local_background_estimator.r_in_scale * psf_params['semimajor_axis']
            annulus_outer_width = 2 * photometer.local_background_estimator.r_out_scale * psf_params['semimajor_axis']
            annulus_inner_height = 2 * photometer.local_background_estimator.r_in_scale * psf_params['semiminor_axis']
            annulus_outer_height = 2 * photometer.local_background_estimator.r_out_scale * psf_params['semiminor_axis']
            
            for coord in np.argwhere(annulus_mask):
                row, col = coord
                
                # offset coords to fit within bbox region
                row += dy
                col += dx
                
                rect = Rectangle((col - 0.5, row - 0.5), 1, 1, linewidth=1, edgecolor='red', facecolor='none')
                circ = Circle((col, row), .1, linewidth=1, edgecolor='red', facecolor='none')
                axes[i].add_patch(rect)
                axes[i].add_patch(circ)
            
            inner_ellipse = Ellipse(centre,
                                    width=annulus_inner_width,
                                    height=annulus_inner_height,
                                    angle=theta,
                                    facecolor='none',
                                    edgecolor='blue',
                                    lw=1,
                                    ls='--',
                                    )
            axes[i].add_patch(inner_ellipse)
            
            outer_ellipse = Ellipse(centre,
                                    width=annulus_outer_width,
                                    height=annulus_outer_height,
                                    angle=theta,
                                    facecolor='none',
                                    edgecolor='blue',
                                    lw=1,
                                    ls='--',
                                    )
            axes[i].add_patch(outer_ellipse)
        
        aperture_ellipse = Ellipse(
            centre,
            width=2 * aperture.a,
            height=2 * aperture.b,
            angle=theta,
            facecolor='none',
            edgecolor='blue',
            lw=1,
            ls='-',
            )
        axes[i].add_patch(aperture_ellipse)
        
        axes[i].set_xlabel('X', fontsize='large')
        axes[i].set_ylabel('Y', fontsize='large')
        axes[i].set_title(f'Source {target}', fontsize='large')
    
    fig.suptitle(key)
    
    if save:
        save_path = out_directory / 'diag' / 'apertures'
        if not save_path.is_dir():
            save_path.mkdir(parents=True)
        save_figure(
            fig=fig,
            path=save_path / f'{key}_apertures.pdf',
            )
    
    if show:
        plt.show(fig)
    else:
        fig.clear()
        plt.close(fig)


def get_max_region_size(
    targets: list[int],
    photometer: AperturePhotometer,
    data: NDArray[np.float64],
    cat: QTable,
    psf_params: dict[str, float],
    ) -> int:
    """
    Get the maximum region size for plotting apertures.
    
    Parameters
    ----------
    targets : list[int]
        The target source IDs.
    photometer : AperturePhotometer
        The `AperturePhotometer` instance.
    data : NDArray[np.float64]
        The image data.
    cat : QTable
        The source catalog.
    psf_params : dict[str, float]
        The PSF parameters.
    
    Returns
    -------
    int
        The maximum region size.
    """
    
    region_sizes = []
    
    for target in targets:
        i = targets.index(target)
        position = [cat['x_centroid'][i], cat['y_centroid'][i]]
        
        aperture = photometer.get_aperture(
            position=position,
            psf_params=psf_params,
            )
        
        if photometer.local_background_estimator is not None:
            annulus_stats = photometer.local_background_estimator.get_stats(
                data=data,
                position=position,
                semimajor_axis=psf_params['semimajor_axis'],
                semiminor_axis=psf_params['semiminor_axis'],
                theta=psf_params['orientation'],
                )
            
            bbox = annulus_stats.bbox
        else:
            aperture_stats = ApertureStats(
                data=data,
                aperture=aperture,
                )
            
            bbox = aperture_stats.bbox
        
        width = bbox.ixmax - bbox.ixmin
        height = bbox.iymax - bbox.iymin
        region_sizes.append(max(width, height))
    
    return max(region_sizes)


def plot_light_curves(
    keys: list[str],
    light_curves: TimeSeries,
    t_ref: Time,
    y_label: Any = None,
    ) -> Figure:
    """
    Plot a table of light curves using a dedicated subplot for each filter.
    
    Parameters
    ----------
    keys : list[str]
        The light curve camera:filter keys.
    light_curves : TimeSeries
        The light curves.
    t_ref : Quantity
        The reference time. Light curves are plotted in seconds from this reference time.
    y_label : Any, optional
        The y-axis label, by default `None`.
    
    Returns
    -------
    Figure
        The resulting figure.
    """
    
    nrows: int = len(keys)
    
    fig, axes = plt.subplots(
        nrows=nrows,
        figsize=(2 * 6.4, .5 * nrows * 4.8),
        tight_layout=True,
        sharex=True,
        gridspec_kw={
            "hspace": 0,
            },
        )
    
    if nrows == 1:
        axes = [axes]
    
    if t_ref is None:
        t_ref = light_curves.time.min()
    
    for i, key in enumerate(keys):
        
        lc = get_lc(light_curves, key=key)
        
        time = (lc.time - t_ref).to_value(u.s)
        flux = lc[f'{key}_rel_flux']
        flux_err = lc[f'{key}_rel_flux_err']
        
        axes[i].errorbar(
            time,
            flux,
            flux_err,
            marker='none',
            linestyle='none',
            ecolor='grey',
            elinewidth=1,
            alpha=.5,
            )
        axes[i].step(
            time,
            flux,
            where='mid',
            lw=1,
            color='k',
            )
        
        axes[i].plot(
                [],
                [],
                marker='none',
                linestyle='none',
                label=key,
            )
        
        axes[i].legend(
            handlelength=0,
            fontsize='x-large',
            frameon=False,
        )
    
    axes[-1].set_xlabel(f'Time from BMJD {t_ref.mjd:.4f} [s]', fontsize='large')
    
    if y_label is not None:
        axes[nrows // 2].set_ylabel(f'{y_label}', fontsize='large')
    
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True)
    
    return fig


