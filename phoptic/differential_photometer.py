import json
from pathlib import Path
from typing import List, Tuple


from astroalign import find_transform
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas import DataFrame


from phoptic.analyzer import Analyzer
from phoptic.utils.fits_handlers import get_stacked_images
from phoptic.utils.helpers import sort_filters
from phoptic.plotting.plots import plot_catalogs, plot_light_curves




class DifferentialPhotometer:
    """
    Helper class for creating relative light curves.
    """


    def __init__(
        self,
        out_directory: Path,
        show_plots: bool = True,
        ) -> None:
        """
        Helper class for creating relative light curves.
        
        Parameters
        ----------
        out_directory : Path
            The path to the directory where output will be saved.
        show_plots : bool, optional
            Whether plots should be shown as they're generated, by default `True`.
        
        Raises
        ------
        FileNotFoundError
            If out_directory cannot be found.
        """
        
        ########################################### input params ###########################################
        
        self.out_directory = Path(out_directory)
        if not self.out_directory.is_dir():
            raise FileNotFoundError(f'[PHOPTIC] {self.out_directory} not found.')
        
        self.show_plots = show_plots
        
        ########################################### attributes ###########################################
        
        with open(self.out_directory.joinpath('misc/reduction_parameters.json'), 'r') as file:
            input_parameters = json.load(file)
        
        self.keys = input_parameters['keys']
        self.time_key = 'BMJD' if input_parameters['barycenter'] else 'MJD'
        self.time_scale = 'tdb' if input_parameters['barycenter'] else 'utc'
        self.t_ref = Time(input_parameters['t_ref'], format='mjd', scale=self.time_scale)
        
        # output filters
        print('[PHOPTIC] Keys: ' + ', '.join(list(self.keys)))
        
        ########################################### read catalogs ###########################################
        
        self.catalogs = {}
        for key in self.keys:
            try:
                cat = QTable.read(
                    self.out_directory.joinpath(f'cat/{key}_catalog.ecsv'),
                    format='ascii.ecsv',
                    )
            except:
                print(f'[PHOPTIC] Could not load {self.out_directory.joinpath(f'cat/{key}_catalog.ecsv')}, skipping.')
                continue
            
            self.catalogs.update({key: cat})
        
        # update keys to match catalog
        self.keys = list(self.catalogs.keys())
        
        ########################################### plot catalogs ###########################################
        
        if show_plots:
            stacked_images = get_stacked_images(self.out_directory)
            
            plot_catalogs(
                out_directory=self.out_directory,
                stacked_images=stacked_images,
                catalogs=self.catalogs,
                show=show_plots,
                save=False,
                )
            
            plt.show()


    def get_relative_light_curve(
        self,
        key: str,
        target: int,
        comparisons: int | List[int],
        phot_label: str,
        prefix: str | None = None,
        match_other_cameras: bool = False,
        show_diagnostics: bool = True,
        ) -> Analyzer:
        """
        Compute the relative light curve for a target source with respect to one or more comparison sources. By default,
        the relative light curve is computed for a single filter. The relative light curve is saved to
        out_directory/relative_light_curves. To automatically match the target and comparison sources across the other
        two filters, set match_other_cameras to True. Note that this can incorrectly match sources, so it is recommended
        to manually check the results.
        
        Parameters
        ----------
        key : str
            The camera:filter key for which the relative light curve will be computed.
        target : int
            The catalog ID of the target source.
        comparisons : int | List[int]
            The catalog ID(s) of the comparison source(s).
        phot_label : str
            The photometry label, used for file reading and labelling.
        prefix : str, optional
            The prefix to use when saving the relative light curve (e.g., the target star's name), by default None.
        match_other_cameras : bool, optional
            Whether to match the target and comparison(s) IDs to the remaining catalog filters, by default `False`. If
            `True`, astroalign must be installed.
        show_diagnostics : bool, optional
            Whether to show diagnostic plots, by default True.
        
        Returns
        -------
        Analyzer
            An Analyzer object containing the relative light curve(s).
        """
        
        if not self.out_directory.joinpath('relative_light_curves').is_dir():
            self.out_directory.joinpath('relative_light_curves').mkdir(parents=True)
        
        # validate filter
        if key not in self.keys:
            raise ValueError(f'[PHOPTIC] {key} is not a valid key.')
        
        # if a single comparison source is given, convert to list
        if isinstance(comparisons, int):
            comparisons = [comparisons]
        
        relative_light_curves = TimeSeries()
        
        if not match_other_cameras:
            # compute relative light curve for single filter
            new_lc = self._compute_relative_light_curve(
                key=key,
                target=target,
                comparisons=comparisons,
                prefix=prefix,
                phot_label=phot_label,
                show_diagnostics=show_diagnostics,
                )
            matched_filters = [key]
        else:
            # compute relative light curves for all available filters
            new_lc, matched_filters = self._match_other_cameras(
                input_key=key,
                input_target=target,
                input_comparisons=comparisons,
                prefix=prefix,
                phot_label=phot_label,
                show_diagnostics=show_diagnostics,
                )
        
        relative_light_curves = vstack([relative_light_curves, new_lc])
        
        if self.show_plots:
            
            plot_light_curves(
                keys=matched_filters,
                light_curves=relative_light_curves,
                t_ref=self.t_ref,
                y_label='Relative flux',
                )
        
        return Analyzer(
            self.out_directory,
            light_curves=relative_light_curves,
            prefix=prefix,
            phot_label=phot_label,
            show_plots=self.show_plots,
            )


    def _compute_relative_light_curve(
        self,
        key: str,
        target: int,
        comparisons: List[int],
        prefix: str | None,
        phot_label: str,
        show_diagnostics: bool,
        ) -> TimeSeries | None:
        """
        Compute the relative light curve for a target source with respect to one or more comparison sources for a given
        filter.
        
        Parameters
        ----------
        key : str
            The camera:filter key.
        target : int
            The catalog ID of the target source.
        comparisons : List[int]
            The catalog ID(s) of the comparison source(s).
        prefix : str | None
            The prefix to use when saving the relative light curve (e.g., the target star's name), by default None.
        phot_label : str
            The photometry label, used for file reading and labelling.
        show_diagnostics : bool
            Whether to show diagnostic plots, by default True.
        
        Returns
        -------
        TimeSeries | None
            The relative light curve for the target source with respect to the comparison sources, or None if the light
            curve could not be computed.
        """
        
        # subdirectory where results will be saved
        light_curve_dir = f'lcs/{phot_label}'
        
        target_df = pd.read_csv(self.out_directory.joinpath(f'{light_curve_dir}/{key}_source_{target}.csv'))
        
        comp_dfs = []
        for comp in comparisons:
            path = self.out_directory.joinpath(f'{light_curve_dir}/{key}_source_{comp}.csv')
            try:
                comparison_df = pd.read_csv(path)
            except:
                print(f'[PHOPTIC] Could not load {path}, skipping ...')
                continue
            comp_dfs.append(comparison_df)
        
        # ensure all light curves have the same time values
        filtered_target_df, filtered_comp_dfs = filter_dataframes_to_common_time_column(
            target_df=target_df,
            comp_dfs=comp_dfs,
            time_key=self.time_key,
            )
        
        # diagnostic plots
        self._plot_diags(
            key=key,
            target=target,
            comparisons=comparisons,
            target_df=filtered_target_df,
            comp_dfs=filtered_comp_dfs,
            phot_label=phot_label,
            show=show_diagnostics,
            )
        
        # get relative light curve
        time, relative_flux, relative_flux_error = compute_relative_flux(
            time=np.asarray(filtered_target_df[self.time_key].values),
            target_df=filtered_target_df,
            comp_dfs=filtered_comp_dfs,
            )
        
        # save relative light curve to CSV
        DataFrame({
            self.time_key: time,
            f'{key}_rel_flux': relative_flux,
            f'{key}_rel_flux_err': relative_flux_error,
        }).to_csv(
            self.out_directory.joinpath(f'relative_light_curves/{phot_label}/{prefix}_{key}_light_curve.csv'),
            index=False,
        )
        
        ts = TimeSeries(time=Time(time, format='mjd', scale=self.time_scale))
        ts[f'{key}_rel_flux'] = relative_flux
        ts[f'{key}_rel_flux_err'] = relative_flux_error
        
        return ts


    def _match_other_cameras(
        self,
        input_key: str,
        input_target: int,
        input_comparisons: List[int],
        prefix: str | None,
        phot_label: str,
        show_diagnostics: bool,
        ) -> Tuple[TimeSeries, List[str]]:
        """
        Compute the relative light curves for all available filters.
        
        Parameters
        ----------
        input_key : str
            The input filter.
        input_target : int
            The target ID in the input filter's catalog.
        input_comparisons : List[int]
            The comparison ID(s) in the input filter's catalog.
        prefix : str | None
            The prefix to use when saving the relative light curve (e.g., the target source's name).
        phot_label : str
            The photometry label.
        show_diagnostics : bool
            Whether to show the diagnostic plots.
        
        Returns
        -------
        Tuple[TimeSeries, List[str]]
            The light curves for all available filters and the list of filters that were successfully matched.
        """
        
        new_lcs = TimeSeries()
        matched_filters = [input_key]
        
        # catalog of input filter
        input_cat = QTable.read(
            self.out_directory.joinpath(f"cat/{input_key}_catalog.ecsv"),
            format="ascii.ecsv",
            )
        
        # source coords in reference filter catalog
        ref_coords = np.asarray([input_cat["x_centroid"].value, input_cat["y_centroid"].value]).T
        ref_target_coords = ref_coords[input_target - 1]  # subtract 1 to account for zero-indexing
        ref_comparison_coords = [ref_coords[comp - 1] for comp in input_comparisons]
        
        for key in self.keys:
            if key == input_key:
                # no matching necessary
                new_lc = self._compute_relative_light_curve(
                    input_key,
                    input_target,
                    input_comparisons,
                    prefix,
                    phot_label,
                    show_diagnostics,
                    )
            else:
                # try to match source positions to new filter
                try:
                    matched_target, matched_comparisons = transform_IDs(
                        self.out_directory,
                        ref_coords,
                        ref_target_coords,
                        ref_comparison_coords,
                        key,
                        )
                    
                    print(f'[PHOPTIC] {input_key} target ID {input_target} was matched to {key} target ID {matched_target}')
                    for i in range(len(input_comparisons)):
                        print(f'[PHOPTIC] {key} comparison ID {input_comparisons[i]} was matched to {key} comparison ID {matched_comparisons[i]}')
                    matched_filters.append(key)
                except:
                    print(f'[PHOPTIC] Could not match {key} sources to {input_key} sources. This can happen if many stars are not identified across all catalogs. Sometimes simply trying again can help (RNG is involved), but often increasing max_catalog_sources in Catalog.create_catalogs() will more reliably solve the issue.')
                    continue
                
                new_lc = self._compute_relative_light_curve(
                    key=key,
                    target=matched_target,
                    comparisons=matched_comparisons,
                    prefix=prefix,
                    phot_label=phot_label,
                    show_diagnostics=show_diagnostics,
                    )
            
            new_lcs = vstack([new_lcs, new_lc])
        
        return new_lcs, sort_filters(matched_filters)


    def _plot_diags(
        self,
        key: str,
        target: int,
        comparisons: List[int],
        target_df: DataFrame,
        comp_dfs: List[DataFrame],
        phot_label: str,
        show: bool,
        ) -> None:
        """
        Plot a combination of diagnostic plots for the specified target and comparison sources.
        
        Parameters
        ----------
        key : str
            The image filter.
        target : int
            The target ID.
        comparisons : List[int]
            The comparison ID(s).
        target_df : DataFrame
            The target light curve.
        comp_dfs : List[DataFrame]
            The comparison light curve(s).
        phot_label : str
            The photometry label.
        show : bool
            Whether to show the plots.
        """
        
        for i, df in enumerate(comp_dfs):
            # diagnostics between target and all comparison sources
            self._plot_diag(
                key=key,
                comparison1=target,
                comparison2=comparisons[i],
                comparison1_df=target_df,
                comparison2_df=df,
                phot_label=phot_label,
                show=show,
            )
            for j, df2 in enumerate(comp_dfs):
                if i != j:
                    # diagnostics between each pair of comparison sources
                    self._plot_diag(
                        key=key,
                        comparison1=comparisons[i],
                        comparison2=comparisons[j],
                        comparison1_df=df,
                        comparison2_df=df2,
                        phot_label=phot_label,
                        show=show,
                        )


    def _plot_diag(
        self,
        key: str,
        comparison1: int,
        comparison2: int,
        comparison1_df: DataFrame,
        comparison2_df: DataFrame,
        phot_label: str,
        show: bool,
        ) -> None:
        """
        Plot the relative diagnostic light curve for two comparison sources for a given filter.
        
        Parameters
        ----------
        key : str
            The filter to compute the relative light curve.
        comparison1 : int
            The catalog ID of the first comparison source.
        comparison2 : int
            The catalog ID of the second comparison source.
        comparison1_df : DataFrame
            The data frame of the first comparison source.
        comparison2_df : DataFrame
            The data frame of the second comparison source.
        phot_label : str
            The photometry label.
        show : bool
            Whether to show the diagnostic plot.
        """
        
        fig, axes = plt.subplots(
            nrows=2,
            tight_layout=True,
            sharex=True,
            figsize=(6.4, 1.5 * 4.8),
            gridspec_kw={
                "hspace": 0,
                },
            )
        
        time, relative_flux, relative_flux_error = compute_relative_flux(
            time=np.asarray(comparison1_df[self.time_key].values),
            target_df=comparison1_df,
            comp_dfs=[comparison2_df],
            )
        
        # convert time to seconds from t_ref
        time = (time  - self.t_ref.value) * 86400
        
        ########################################### normalised light curves ###########################################
        
        axes[0].set_title(f'{key} Target ID: {comparison1}, Comparison ID: {comparison2}')
        axes[0].errorbar(
            (comparison1_df[self.time_key].values - self.t_ref.mjd) * 86400,
            comparison1_df['flux'] / comparison1_df['flux'].median(),
            np.abs(comparison1_df['flux_err'] / comparison1_df['flux'].median()),
            fmt="kx-",
            ms=5,
            elinewidth=1,
            label=f'Source {comparison1}',
            alpha=.5,
            )
        axes[0].errorbar(
            (comparison2_df[self.time_key].values - self.t_ref.mjd) * 86400,
            comparison2_df['flux'] / comparison2_df['flux'].median(),
            np.abs(comparison2_df['flux_err'] / comparison2_df['flux'].median()),
            fmt="r+-",
            ms=5,
            elinewidth=1,
            label=f'Source {comparison2}',
            alpha=.5,
            )
        axes[0].legend()
        axes[0].set_ylabel("Normalized raw flux")
        
        ########################################### relative light curve ###########################################
        
        axes[1].errorbar(
            time,
            relative_flux / np.median(relative_flux),
            relative_flux_error / np.abs(np.median(relative_flux)),
            fmt="k.",
            ms=2,
            ecolor="grey",
            elinewidth=1,
            )
        axes[1].axhline(
            1,
            color='r',
            lw=1,
        )
        axes[1].set_xlabel(f"Time from {self.time_key} {self.t_ref.value:.4f} [s]")
        axes[1].set_ylabel("Normalized relative flux")
        
        ########################################### format plot ###########################################
        
        for ax in axes:
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
        
        ########################################### save plot ###########################################
        
        save_dir = self.out_directory.joinpath(f'relative_light_curves/{phot_label}/diag')
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True)
        
        fig.savefig(save_dir.joinpath(f'{key}_{comparison1}_{comparison2}_diag_light_curve.png'))
        
        ########################################### optionally show plot ###########################################
        
        if not show:
            fig.clear()
            plt.close(fig)




def filter_dataframes_to_common_time_column(
    target_df: DataFrame,
    comp_dfs: List[DataFrame],
    time_key: str,
    ) -> Tuple[DataFrame, List[DataFrame]]:
    """
    Get the matching times between a target data frame (light curve) and a list of comparison data frames (light 
    curves).
    
    Parameters
    ----------
    target_df : DataFrame
        The data frame of the target source.
    comp_dfs : List[DataFrame]
        The list of data frames of the comparison sources.
    time_key : str,
        The time key (either BMJD or MJD depending on whether Barycentric corrections were applied).
    
    Returns
    -------
    Tuple[DataFrame, List[DataFrame]]
        The filtered target data frame and the list of filtered comparison data frames.
    """
    
    # get time columns from all data frames
    time_columns = [target_df[time_key].values]
    time_columns.extend([df[time_key].values for df in comp_dfs])
    
    # get matching times between all data frames
    common_times = set(time_columns[0])
    for time_col in time_columns[1:]:
        common_times.intersection_update(time_col)
    common_times = sorted(common_times)
    
    # get matching times for target
    filtered_target_df = target_df[target_df[time_key].isin(common_times)]
    filtered_target_df.reset_index(drop=True, inplace=True)
    
    # get matching times for comparisons
    filtered_comp_dfs = [df[df[time_key].isin(common_times)] for df in comp_dfs]
    filtered_comp_dfs = [df.reset_index(drop=True) for df in filtered_comp_dfs]
    
    return filtered_target_df, filtered_comp_dfs


def compute_relative_flux(
    time: NDArray[np.float64],
    target_df: DataFrame,
    comp_dfs: List[DataFrame],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute the relative flux between a target source and one or more comparison sources.
    
    Parameters
    ----------
    time : NDArray[np.float64]
        The time column.
    target_df : DataFrame
        The light curve of the target source.
    comp_dfs : List[DataFrame]
        The light curves of the comparison sources.
    
    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        The time, relative flux, and relative flux error.
    """
    
    comp_fluxes = np.sum(np.asarray([df['flux'].values for df in comp_dfs]), axis=0)
    comp_flux_errors = np.sqrt(np.sum([np.square(df['flux_err'].values) for df in comp_dfs], axis=0))
    
    target_flux = target_df['flux'].values
    target_flux_err = target_df['flux_err'].values
    
    relative_flux = target_flux / comp_fluxes
    relative_flux_error = relative_flux * np.sqrt(np.square(target_flux_err / target_flux) + np.square(comp_flux_errors / comp_fluxes))
    
    return time, relative_flux, np.abs(relative_flux_error)


def transform_IDs(
    out_directory: Path,
    ref_coords: NDArray,
    ref_target_coords: NDArray,
    ref_comparison_coords: List[NDArray],
    key: str,
    ) -> Tuple[int, List[int]]:
    """
    Transform some source IDs from one camera to another.
    
    Parameters
    ----------
    out_directory : Path
        The output directory.
    ref_coords : NDArray
        The coordinates of all sources in the current (reference) catalogue.
    ref_target_coords : NDArray
        The coordinates of the sources being transformed in the current (reference) catalogue.
    ref_comparison_coords : List[NDArray]
        The coordiantes of all sources in the new (comparison) catalogue.
    key : str
        The current filter.
    
    Returns
    -------
    Tuple[int, List[int]]
        The transformed target and comparison source IDs.
    """
    
    # get source positions in new filter
    cat = QTable.read(
        out_directory.joinpath(f"cat/{key}_catalog.ecsv"),
        format="ascii.ecsv",
        )
    coords = np.asarray([cat["x_centroid"].value, cat["y_centroid"].value]).T
    
    # get star-to-star correspondence
    source_arr, ref_arr = find_transform(coords, ref_coords)[1]
    
    # get transformed coordinates for target and comparison(s)
    target_coords = source_arr[np.where(ref_arr == ref_target_coords)]
    comparison_coords = [source_arr[np.where(ref_arr == comp_coords)] for comp_coords in ref_comparison_coords]
    
    # get transformed IDs for target and comparison(s)
    new_target = int(np.where(coords == target_coords)[0][0]) + 1
    new_comparisons = [int(np.where(coords == comp_coords)[0][0]) + 1 for comp_coords in comparison_coords]
    
    return new_target, new_comparisons






