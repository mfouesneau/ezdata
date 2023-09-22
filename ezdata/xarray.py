""" Xarray and Pandas binning tools """
import xarray
import numpy as np
import pandas as pd
from typing import Union, Sequence, Tuple, Callable
from scipy.stats import binned_statistic_dd
import xarray as xr

# plotting tools
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from numpy.typing import NDArray, ArrayLike
except ImportError:
    # older numpy version
    NDArray = "NDArray"
    ArrayLike = "ArrayLike"

Scalar = np.number


def guess_bins(x: Union[ArrayLike, pd.Series]) -> NDArray:
    """find the smallest diff(x) and set bins to min(x):max(x):min(diff(x))"""
    x_ = np.sort(np.unique(x))
    dx_ = np.diff(x_).min()
    coo = np.arange(
        x_.min() - 0.5 * dx_, x_.max() + 0.6 * dx_, dx_  # to ensure the last bin!
    )
    return coo


def guess_range(x: Union[ArrayLike, pd.Series]) -> NDArray:
    """Guess the value range of the data (1D or N-D)"""
    return np.array([np.nanmin(x, 0), np.nanmax(x, 0)]).T


def xr_histogram(
    values: ArrayLike,
    bins: Union[int, Sequence[Scalar], str] = None,
    name: str = None,
    binby: Sequence[str] = None,
    range: Sequence[Tuple[float, float]] = None,
    **kwargs
) -> xarray.DataArray:
    """Return an np.histogramdd as xarray.DataArray

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
        binby = ["x{0:d}".format(k) for k in np.arange(len(coords))]
    array = xarray.DataArray(N, coords=list(zip(binby, coords)))
    array.name = name
    return array


def xr_histogram_df(
    df: pd.DataFrame,
    binby: Sequence[str],
    bins: Union[int, Sequence[Scalar], str] = None,
    range: Sequence[Tuple[float, float]] = None,
    **kwargs
) -> xarray.Dataset:
    """Run the xr_histogram on all other variables than binby
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
        [
            xr_histogram(
                np_data,
                bins=bins,
                weights=df[k],
                name=k,
                binby=binby,
                range=range,
                **kwargs
            )
            for k in df.columns
            if k not in binby
        ]
    )


def df_groupby_bins(
    data: pd.DataFrame,
    binby: Sequence[str],
    bins: Union[int, Sequence[Scalar], str] = None,
    agg: Union[str, Callable, Sequence, dict] = "sum",
) -> xarray.Dataset:
    """Group data into bins while keeping empty bins.
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

    r = (
        data.groupby(binby)
        .agg(agg)
        .reindex(pd.MultiIndex.from_product(coords, names=binby))
    )
    return r.to_xarray()


def centers_to_bin_edges(x):
    """Return bin egdes from centers given by x"""
    x = np.array(x)
    dx = np.diff(x)
    return np.hstack((x[:-1] - 0.5 * dx, x[-1] + 0.5 * dx[-1]))


def xr_histogram_like(
    ref_xr: Union[xarray.Dataset, xarray.DataArray],
    binby: Sequence[str] = None,
    **values
) -> xarray.DataArray:
    """Make an histogram copying the bins from a reference dataset

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
        binby = [k for k in list(ref_xr.coords.keys()) if k in values]
    kwargs = {k: v for k, v in values.items() if k not in binby}

    coords = {k: ref_xr.coords[k].values for k in binby}
    bins = [centers_to_bin_edges(coords[k]) for k in binby]

    X_ = np.array([values[k] for k in binby]).T
    return xr_histogram(X_, bins=bins, binby=binby, **kwargs)


def xr_binned_statistic_df(
    df: pd.DataFrame,
    binby: Sequence[str],
    bins: Union[int, Sequence[np.number], str] = None,
    range: Union[Sequence[np.number], np.number] = None,
    statistic: Union[Callable, str] = "mean",
    **kwargs
) -> xr.Dataset:
    """Create a binned dataset from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        data to be binned
    binby : Sequence[str]
        dimensions to bin by
    bins : Union[int, Sequence[np.number], str] = None
        number of bins or bin edges
    range : Union[Sequence[np.number], np.number] = None
        limits of the bins to be used
    statistic : Union[Callable, str] = "mean"
        which statistic to use for aggregating the data per bin
    **kwargs : dict

    Returns
    -------
    xr.Dataset
       xr.Dataset containing the binned data.
    """
    if bins is None:
        bins = []
        for k in binby:
            if isinstance(df[k].dtype, pd.CategoricalDtype):
                # Need to deal with categorical data differently
                bins.append(guess_bins(df[k].cat.codes.dropna()))
            else:
                bins.append(guess_bins(df[k].dropna()))

    np_data = df[binby].copy()
    # Binning on numerial values only,
    # i.e. replace categorical values with their codes
    for k in binby:
        if isinstance(df[k].dtype, pd.CategoricalDtype):
            np_data[k] = df[k].cat.codes
    np_data = np_data.to_numpy()
    if range is None:
        range = guess_range(np_data)

    other_keys = [k for k in df.columns if k not in binby]
    values = df[other_keys]

    res = binned_statistic_dd(
        np_data, values.T, statistic=statistic, bins=bins, range=range
    )

    coords = [0.5 * (coo[:-1] + coo[1:]) for coo in res.bin_edges]
    # Correct for categorical data: replace codes with categories
    for e, k in enumerate(binby):
        if isinstance(df[k].dtype, pd.CategoricalDtype):
            coords[e] = df[k].cat.categories[coords[e].astype(int)]

    data = [
        xr.DataArray(vals, coords=list(zip(binby, coords))) for vals in res.statistic
    ]
    for dk, name in zip(data, other_keys):
        dk.name = name
    return xr.merge(data)


def get_bins_from_xr(
    ref_xr: Union[xr.Dataset, xr.DataArray], binby: Sequence[str] = None
) -> dict:
    """Returns the bin edges from coordinates of an xarray Dataset or DataArray.

    Parameters
    ----------
    ref_xr : Union[xr.Dataset, xr.DataArray]
        reference xarray Dataset or DataArray.
    binby : Sequence[str], optional
        named dimensions to consider. The default is all from the reference xarray Dataset or DataArray.

    Returns
    -------
    bin_edges : dict
        dictionary of bin edges per named coordinates.
    """
    if binby is None:
        binby = [k for k in list(ref_xr.coords.keys())]
    coords = {k: ref_xr.coords[k].values for k in binby}
    bins = [centers_to_bin_edges(coords[k]) for k in binby]
    return dict(zip(binby, bins))


def image_ops(ops: Callable, da: xr.DataArray, *args, **kwargs) -> xr.DataArray:
    """Apply an image operation function to an xarray DataArray.

    Parameters
    ----------
    ops : Callable
        image operation function.
    da : xr.DataArray
        image to be operated on.
    *args, **kwargs
        arguments to be passed to the image operation function.

    Returns
    -------
    xr.DataArray
        operated image.
    """
    newdr = da.copy(deep=True)
    newdr.data = ops(newdr.data, *args, **kwargs)
    return newdr


def get_edges_from_coords(coords: np.ndarray) -> np.ndarray:
    """Return edges from coords

    Parameters
    ----------
    coords : np.ndarray
        coordinates of bin centers

    Returns
    -------
    edges : np.ndarray
        bin edges
    """
    delta = np.diff(coords)
    return np.hstack(
        [
            coords[0] - 0.5 * delta[0],
            coords[:-1] + 0.5 * delta,
            coords[-1] + 0.5 * delta[-1],
        ]
    )


def plot_voxels(
    da: xr.DataArray,
    ax: plt.Axes = None,
    cmap: Union[str, mpl.colors.Colormap] = "viridis",
    add_colorbar: bool = True,
    alpha: float = 0.9,
    norm: Union[str, mpl.colors.Normalize] = None,
    vmin: float = None,
    vmax: float = None,
    **kwargs
) -> dict:
    """Plot a 3D voxel plot of an xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray
        3D data array to be plotted as voxels.
    ax : plt.Axes, optional
        matplotlib axes to plot on.
        The default is None (new 3d projected axis will be created).
    cmap : Union[str, mpl.colors.Colormap], optional
        colormap to use. The default is "viridis".
    add_colorbar : bool, optional
        whether to add a colorbar. The default is True.
    alpha : float, optional
        alpha value for the voxels. The default is 0.9.
    norm : Union[str, mpl.colors.Normalize], optional
        The normalization method used to scale scalar data to the [0, 1] range
        before mapping to colors using *cmap*. By default, a linear scaling is
        used, mapping the lowest value to 0 and the highest to 1.

        If given, this can be one of the following:

        - An instance of `.Normalize` or one of its subclasses
        (see :doc:`/tutorials/colors/colormapnorms`).
        - A scale name, i.e. one of "linear", "log", "symlog", "logit", etc.  For a
        list of available scales, call `matplotlib.scale.get_scale_names()`.
        In that case, a suitable `.Normalize` subclass is dynamically generated
        and instantiated.
    vmin : float, optional
        the minimum data value that the colormap covers.
    vmax : float, optional
        the maximum data value that the colormap covers.
    **kwargs : dict
        additional keyword arguments to be passed to the matplotlib voxels function.

    Returns
    -------
    voxels : dict
        dictionary of voxels: {(i, j, k) bins: mpl_toolkits.mplot3d.art3d.Poly3DCollection}.
    """
    from mpl_toolkits.mplot3d import axes3d  # noqa: F401 unused import

    if ax is None:
        ax = plt.subplot(111, projection="3d")

    # voxels requires edges not centers
    bins = [get_edges_from_coords(da.coords[dim]) for dim in da.dims]

    cube = da.to_numpy()
    notmask = np.isfinite(cube)

    # set vmin, vmax, and norm
    if vmin is None:
        vmin = np.nanmin(cube)
    if vmax is None:
        vmax = np.nanmax(cube)
    if norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # convert data values to colors
    palette = mpl.colormaps.get_cmap(cmap)
    carr = palette(norm(cube))
    carr[:, :, :, 3] = alpha

    # plot
    X, Y, Z = np.meshgrid(*bins, indexing="ij")
    voxels = ax.voxels(X, Y, Z, notmask, facecolors=carr, **kwargs)

    # set default labels and title
    axis_labels = np.array([xr.plot.utils.label_from_attrs(da[dim]) for dim in da.dims])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(da._title_for_slice())

    # add colorbar
    if add_colorbar:
        m = mpl.cm.ScalarMappable(cmap=palette, norm=norm)
        m.set_array([])
        plt.gcf().colorbar(m, ax=ax)

    return voxels
