from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries
from astropy.units import Quantity
import numpy as np
from numpy.typing import NDArray




def get_lc(
    light_curves: TimeSeries,
    key: str,
    ) -> TimeSeries:
    """
    Given a table of light curves, extract the light curve for a single key.
    
    Parameters
    ----------
    light_curves : TimeSeries
        The table of light curves.
    key : str
        The camera:filter key (e.g., "1:g" for camera 1 with a g filter).
    
    Returns
    -------
    TimeSeries
        The light curve for the filter.
    """
    
    flux_col = f'{key}_rel_flux'
    err_col = f'{key}_rel_flux_err'
    
    time_arr = light_curves['time']
    flux_arr = light_curves[flux_col]
    err_arr = light_curves[err_col]
    
    if hasattr(flux_arr, 'mask') and hasattr(err_arr, 'mask'):
        mask = flux_arr.mask | err_arr.mask
    else:
        mask = ~(np.isfinite(flux_arr) & np.isfinite(err_arr))
    
    new_lc = TimeSeries(time=time_arr[~mask])
    new_lc[flux_col] = flux_arr[~mask]
    new_lc[err_col] = err_arr[~mask]
    
    return new_lc


def split_timeseries_on_gaps(
    ts: TimeSeries,
    threshold: float = 1.5,
    ) -> list[TimeSeries]:
    """
    Split a time series into a list of contiguous time series. Similar to `stingray`'s `split_by_gti()` method.
    
    Parameters
    ----------
    ts : TimeSeries
        The time series, assumed to contain gaps.
    threshold : float, optional
        The gap detection threshold, by default 1.5 times the median time delta.
    
    Returns
    -------
    list[TimeSeries]
        The list of strictly contiguous time series.
    """
    
    ts_list = []
    
    dt = np.diff(ts['time'].mjd)  # convert to MJD to remove units
    gap_indices = np.where(dt > threshold * np.median(dt))[0] + 1  # add 1 to shift index to first point after gap
    edge_indices = np.concat(([0], gap_indices, [len(ts)]))  # include start and end of time series
    
    for i in range(1, len(edge_indices)):
        ts_list.append(ts[edge_indices[i-1]:edge_indices[i]])
    
    return ts_list


def segment_timeseries(
    ts: TimeSeries,
    segment_size: Quantity,
    ) -> list[TimeSeries]:
    """
    Split a time series into equal length, contiguous segments.
    
    Parameters
    ----------
    ts : TimeSeries
        The time series.
    segment_size : Quantity
        The segment size.
    
    Returns
    -------
    list[TimeSeries]
        The time series segments.
    
    Raises
    ------
    ValueError
        If no valid segments could be found.
    """
    
    dt = float(np.median(np.diff(ts.time.mjd)))  # nominal time resolution
    
    segments: list[TimeSeries] = []
    
    # split time series on gaps
    ts_list = split_timeseries_on_gaps(ts)
    
    n: int = get_segment_size(
        dt=dt,
        segment_size=segment_size,
        )
    
    for ts in ts_list:
        prev = 0
        while prev + n <= len(ts):
            segments.append(ts[prev:prev + n])
            prev += n
    
    if len(segments) == 0:
        raise ValueError(f'[PHOPTIC] No valid segments were found in the input. Consider reducing segment_size.')
    
    return segments



def get_segment_size(
    dt: float,
    segment_size: Quantity,
    ) -> int:
    """
    Get the number of time series rows per segment.
    
    Parameters
    ----------
    dt : float
        The nominal time resolution of the time series in days.
    segment_size : Quantity
        The segment size.
    
    Returns
    -------
    int
        The number of rows per segment.
    """
    
    return round(segment_size.to_value(u.day) / dt)


def infer_gtis(
    time: NDArray | Time | Quantity,
    threshold: float = 1.5,
    ) -> NDArray:
    """
    Infer the Good Time Intervals from a time array.
    
    Parameters
    ----------
    time : NDArray | Time | Quantity
        The time array. If this array has units, the resulting GTIs will have the same units.
    threshold : float, optional
        The gap detection threshold, by default 1.5 times the minimum time delta.
    
    Returns
    -------
    NDArray
        The inferred GTIs.
    """
    
    time = np.sort(time)  # ensure time stamps are sorted
    
    # nominal time resolution
    dt = np.median(np.diff(time))
    
    # compute the gap threshold
    gap_threshold = threshold * dt
    
    # define GTI starts and stops
    gti_starts = [time[0] - dt / 2]
    gti_stops = []
    
    # compute GTIs
    for i in range(1, time.size):
        if time[i] - time[i - 1] > gap_threshold:
            gti_stops.append(time[i - 1] + dt / 2)
            gti_starts.append(time[i] - dt / 2)
    
    if gti_starts[-1] == time[-1]:
        gti_starts.pop()
    else:
        gti_stops.append(time[-1] + dt / 2)
    
    # define GTIs in stingray format
    return np.array(list(zip(gti_starts, gti_stops)))


def uniformly_sampled(
    time: NDArray,
    dt: float,
    raise_error: bool = False,
    ) -> np.bool:
    """
    Check if a time array is uniformly sampled.
    
    Parameters
    ----------
    time : NDArray
        The time array (assumed to be in units of seconds).
    dt : float
        The nominal time resolution (assumed to be in units of seconds).
    raise_error : bool, optional
        Whether to raise an error if the time array is not uniformly sampled, by default `False`.
    
    Returns
    -------
    np.bool
        Whether the time array is uniformly sampled.
    
    Raises
    ------
    ValueError
        If the time array is not uniformly sampled and `raise_error=True`.
    """
    
    
    empirical_dt: NDArray = time[1:] - time[:-1]
    mask = np.isclose(dt, empirical_dt, rtol=1e-6, atol=0.)
    times_match = np.all(mask)
    
    if not times_match and raise_error:
        indices = np.where(~mask)[0]
        
        time_differences = []
        for index in indices:
            time_differences.append(str(empirical_dt[index]))
        
        raise ValueError(f'[PHOPTIC] Irregularly sampled inputs detected.\
            Time resolution: {dt}, but found time differences of {','.join(time_differences)}')
    
    return times_match



def segment_arr(
    t: NDArray,
    y: NDArray,
    segment_size: float,
    y2: NDArray | None = None,
    tolerance: float = 1.5,
    ) -> list[tuple]:
    """
    Segment arrays into uniform segments.
    
    Parameters
    ----------
    t : NDArray
        The time array (in units of seconds).
    y : NDArray
        The signal array.
    segment_size : float
        The desired time-span of the resulting segments.
    
    Returns
    -------
    list[tuple[NDArray, NDArray]]
        The segmented arrays [(t_segment_0, y_segment_0), (t_segment_1, y_segment_1), etc.]
    """
    
    diffs = np.diff(t)
    dt = np.median(diffs)
    seg_n = round(segment_size / dt)
    
    gap_indices = np.flatnonzero(np.abs(diffs > (tolerance * dt))) + 1  # indices of points AFTER gaps
    
    t_chunks = np.split(t, gap_indices)
    y_chunks = np.split(y, gap_indices)
    if y2 is not None:
        y2_chunks = np.split(y2, gap_indices)
    
    segments = []
    for i in range(len(t_chunks)):
        n = len(t_chunks[i])
        if n < seg_n:
            continue
        n_segments = n // seg_n
        for j in range(n_segments):
            t_seg = t_chunks[i][j * seg_n:(j + 1) * seg_n]
            y_seg = y_chunks[i][j * seg_n:(j + 1) * seg_n]
            if y2 is not None:
                y2_seg = y2_chunks[i][j * seg_n:(j + 1) * seg_n]
                segments.append((t_seg, y_seg, y2_seg))
            else:
                segments.append((t_seg, y_seg))
    
    return segments
