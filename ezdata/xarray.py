""" Xarray and Pandas binning tools """
import xarray
from typing import Union, Sequence, Tuple, Callable
try:
    from numpy.typing import NDArray, ArrayLike
except ImportError:
    # older numpy version
    NDArray = "NDArray"
    ArrayLike = "ArrayLike"
import numpy as np
import pandas as pd
Scalar = np.number


def guess_bins(
    x: Union[ArrayLike, pd.Series]
    ) -> NDArray:
    """ find the smallest diff(x) and set bins to min(x):max(x):min(diff(x)) """ 
    x_ = np.sort(np.unique(x))
    dx_ = np.diff(x_).min()
    coo = np.arange(x_.min() - 0.5 * dx_, 
                    x_.max() + 0.5 * dx_, 
                    dx_)
    return coo

  
def guess_range(
    x: Union[ArrayLike, pd.Series]
    ) -> NDArray:
    """ Guess the value range of the data (1D or N-D) """
    return np.array([np.nanmin(x, 0), 
                     np.nanmax(x, 0)]).T

  
def xr_histogram(values: ArrayLike, 
                 bins: Union[int, Sequence[Scalar], str] = None, 
                 name: str = None, 
                 binby: Sequence[str] = None,
                 range: Sequence[Tuple[float, float]] = None, 
                 **kwargs) -> xarray.DataArray:
    """ Return an np.histogramdd as xarray.DataArray 

    Parameters
    ----------
    values : ArrayLike
        Input data. The histogram is computed over the flattened array.

    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.

        .. versionadded:: 1.11.0

        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.
    
    name: str
        name of the returned variable array
    
    binby: Sequence[str]
        names of the binning axes

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.

    kwargs: dict
        send to `np.histogramdd`
    
    returns
    -------
    binned_data: xarray.DataArray
        The binned data as a DataArray object
        It contains all the information as a labeled array
    """
    if range is None:
        range = guess_range(values)
    N, bins = np.histogramdd(values, bins=bins, range=range, **kwargs)
    coords = [0.5 * (coo[:-1] + coo[1:]) for coo in bins]
    if binby is None:
        binby = ['x{0:d}'.format(k) for k in np.arange(len(coords))]
    array = xarray.DataArray(N, coords = list(zip(binby, coords)))
    array.name = name
    return array

  
def xr_histogram_df(
    df: pd.DataFrame, 
    binby: Sequence[str],
    bins: Union[int, Sequence[Scalar], str] = None, 
    range: Sequence[Tuple[float, float]] = None, 
    **kwargs) -> xarray.Dataset:
    """ Run the xr_histogram on all other variables than binby 
    Parameters
    ----------
    df : pd.DataFrame
        Input data. The histogram is computed over the flattened array.
    
    binby: Sequence[str]
        which fields to use as binning dimensions

    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.

        .. versionadded:: 1.11.0

        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.

    range : (float, float), optional
        The lower and upper range of the bins.  If not provided, range
        is simply ``(a.min(), a.max())``.  Values outside the range are
        ignored. The first element of the range must be less than or
        equal to the second. `range` affects the automatic bin
        computation as well. While bin width is computed to be optimal
        based on the actual data within `range`, the bin count will fill
        the entire range including portions containing no data.

    kwargs: dict
        send to `np.histogramdd`
    
    returns
    -------
    binned_dataset: xarray.DataSet
        All the variables apart from binby as a labeled dataset
    """
    np_data = df[binby].to_numpy()

    if bins is None:
        bins = [guess_bins(df[k].dropna()) for k in binby]

    if range is None:
        range = guess_range(np_data)

    return xarray.merge(
        [xr_histogram(np_data, 
                      bins=bins, weights=df[k], 
                      name=k, binby=binby,
                      range=range, **kwargs) 
         for k in df.columns if k not in binby])
    

def df_groupby_bins(
    data: pd.DataFrame, 
    binby: Sequence[str],
    bins: Union[int, Sequence[Scalar], str] = None,
    agg: Union[str, Callable, Sequence, dict] = 'sum',  
    ) -> xarray.Dataset:
    """ Group data into bins while keeping empty bins.
        This function is slower than xr_histogram_df but 
        it is slightly more flexible as it uses pandas 
        grouping and aggretating method.

    Parameters
    ----------
    data: pd.DataFrame
        Data to bin
    binby: Sequence[str]
        which fields to use as binning dimensions
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range (10, by default). If `bins` is a
        sequence, it defines a monotonically increasing array of bin edges,
        including the rightmost edge, allowing for non-uniform bin widths.

        .. versionadded:: 1.11.0

        If `bins` is a string, it defines the method used to calculate the
        optimal bin width, as defined by `histogram_bin_edges`.

    agg: function, str, list or dict
        Function to use for aggregating the data. If a function, must either
        work when passed a DataFrame or when passed to DataFrame.apply.

        Accepted combinations are:

        - function
        - string function name
        - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
        - dict of axis labels -> functions, function names or list of such.
   
    returns
    -------
    binned_dataset: xarray.DataSet
        All the variables apart from binby as a labeled dataset
    """
    if bins is None:
        bins = [guess_bins(data[k].dropna()) for k in binby]
        
    coords = [0.5 * (coo[:-1] + coo[1:]) for coo in bins]

    r = data.groupby(binby)\
            .agg(agg)\
            .reindex(pd.MultiIndex.from_product(coords, names=binby))
    return r.to_xarray()


def centers_to_bin_edges(x):
    """ Return bin egdes from centers given by x """
    x = np.array(x)
    dx = np.diff(x)
    return np.hstack((x[:-1] - 0.5 * dx, x[-1] + 0.5 * dx[-1]))


def xr_histogram_like(ref_xr: Union[xarray.Dataset, xarray.DataArray], 
                      binby: Sequence[str] = None, 
                      **values) -> xarray.DataArray:
    """ Make an histogram copying the bins from a reference dataset 

    Parameters
    ----------
    ref_xr: Union[xarray.Dataset, xarray.DataArray]
        a reference dataset or dataarray to mimic

    binby: Sequence[str]
        names of the binning axes
    
    values: Dict
        values to consider to bin. Names should match the reference dimensions.

    returns
    -------
    binned_data: xarray.DataArray
        The binned data as a DataArray object
        It contains all the information as a labeled array
    """
    if binby is None:
        binby = [k for k in list(ref_xr.coords.keys()) 
                 if k in values]
    kwargs = {k: v for k, v in values.items() if k not in binby}
    
    coords = {k:ref_xr.coords[k].values for k in binby}
    bins = [centers_to_bin_edges(coords[k]) for k in binby]

    X_ = np.array([values[k] for k in binby]).T
    return xr_histogram(X_, bins=bins, binby=binby, **kwargs)
