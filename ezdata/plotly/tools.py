"""Plotly tools"""

from typing import Generator, Optional

import numpy as np

import plotly
import plotly.graph_objects
from plotly.exceptions import PlotlyKeyError


def iter_coloraxis_names(max: int = 100) -> Generator:
    """Iterator that yields useful coloraxis names (i.e. coloraxis, coloraxis2, coloraxis3, ...)"""
    for i in range(1, max + 1):
        yield f"coloraxis{i}" if i > 1 else "coloraxis"


def reposition_colorbars(
    fig: plotly.graph_objects.Figure, xnorm: float = 1.0, ynorm: float = 0.5, **kwargs
) -> plotly.graph_objects.Figure:
    """Reposition colorbars of a figure to the same relative position of each subplot's domain

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        A figure object with subplots
    xnorm : float, optional
        Normalized x position of the colorbar, by default 1.
    ynorm : float, optional
        Normalized y position of the colorbar, by default 0.5
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the colorbar layout update

    Returns
    -------
    plotly.graph_objects.Figure
        The updated figure
    """
    grid_ref = fig._validate_get_grid_ref()
    ncols = len(grid_ref)
    nrows = len(grid_ref[0])

    # flat list of axes
    axes = [
        fig.get_subplot(j, i) for j in range(1, ncols + 1) for i in range(1, nrows + 1)
    ]

    updates = {}
    for ax, coloraxis in zip(axes, iter_coloraxis_names()):
        xpos = ax.xaxis.domain[0] + (ax.xaxis.domain[1] - ax.xaxis.domain[0]) * xnorm
        ypos = ax.yaxis.domain[0] + (ax.yaxis.domain[1] - ax.yaxis.domain[0]) * ynorm
        updates[coloraxis] = {"colorbar": {"x": xpos, "y": ypos, **kwargs}}
    return fig.update_layout(**updates)


def separate_colorbars(
    fig: plotly.graph_objects.Figure, xnorm: float = 1.0, ynorm: float = 0.5, **kwargs
) -> plotly.graph_objects.Figure:
    """Separate colorbars of a figure with subplots and sets them with each subplot (`reposition_colorbars`)
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        A figure object with subplots
    xnorm : float, optional
        Normalized x position of the colorbar, by default 1.
    ynorm : float, optional
        Normalized y position of the colorbar, by default 0.5
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the colorbar layout update

    Returns
    -------
    plotly.graph_objects.Figure
        The updated figure
    """
    fig = fig.update_layout(coloraxis=dict(showscale=True))
    for num, trace in enumerate(fig.data, 1):
        try:
            trace.update({"coloraxis": f"coloraxis{num}" if num > 1 else "coloraxis"})
        except ValueError:
            trace.update({"marker_coloraxis": f"coloraxis{num}" if num > 1 else "coloraxis"})
    fig = reposition_colorbars(fig, xnorm=xnorm, ynorm=ynorm, **kwargs)
    return fig


def get_colorbars(fig: plotly.graph_objects.Figure) -> Generator:
    """Get the colorbars of a figure
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        A figure object
    Returns
    -------
    list
        List of colorbar dictionaries
    """
    for trace in fig.data:
        try:
            cb = fig.layout[trace['coloraxis']]
        except PlotlyKeyError:
            cb = fig.layout[trace['marker_coloraxis']]
        yield cb


def logscale(
    trace: plotly.basedatatypes.BaseTraceType,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    numticks: Optional[int] = None,
    **kwargs,
):
    """
    Apply logarithmic scaling to the data in the trace and update the associated figure (colorbar etc).
    The trace will be updated in place to contain the scaled data and an updated hovertemplate.

    parameters
    ----------
    trace : plotly.graph_objects.Figure
        The trace containing the data to be scaled.
    zmin : float, optional
        The minimum value for the scaled data. If not provided, the minimum value of the original data is used.
        Fixes the minimum value of the colorbar as well.
    zmax : float, optional
        The maximum value for the scaled data. If not provided, the maximum value of the original data is used.
        Fixes the maximum value of the colorbar as well.
    numticks : int, optional
        The number of ticks to be displayed on the colorbar. If not provided, 9 ticks are displayed.
        Note that number does not seem to affect the plot, plotly decides the number of ticks anyways.
    **kwargs : dict
        Additional keyword arguments to be added or forced in the arguments to `update_coloraxes`.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The updated figure with the logarithmically scaled data and colorbar applied.

    Examples
    --------
    >>> fig = px.imshow(np.random.rand(10, 10) ** 2)
    >>> fig = logscale(fig.data[-1], zmin=1e-3, zmax=1)
    """

    # apply data transformation to the z data with overflow protection
    logz = trace.z.copy()
    cond = np.isfinite(trace.z) & (trace.z > 0)
    logz[cond] = np.log10(trace.z[cond])
    logz[~cond] = np.nan

    # get the minimum and maximum values of the transformed data
    logzmin_ = max(np.nanmin(logz), 0)
    logzmax_ = np.nanmax(logz)
    adds = {"cmin": logzmin_, "cmax": logzmax_}
    if zmin is not None:
        logzmin_ = max(np.log10(zmin), logzmin_)
        adds["cmin"] = np.log10(zmin)
    if zmax is not None:
        logzmax_ = min(np.log10(zmax), logzmax_)
        adds["cmax"] = np.log10(zmax)

    if numticks is not None:
        numticks = np.clip(numticks, 2, 9)
    else:
        numticks = 9

    numdec = np.floor(logzmax_) - np.ceil(logzmin_)

    # Get decades between major ticks.
    stride = numdec // numticks + 1

    # if the stride is as big or bigger than the range, clip it to
    # the available range - 1 with a floor of 1.
    if stride >= numdec:
        stride = max(1, numdec - 1)

    decades = np.arange(
        np.floor(logzmin_) - stride, np.ceil(logzmax_) + 2 * stride, stride
    )
    # decades = decades[(decades >= np.floor(logzmin_)) & (decades <= np.ceil(logzmax_))]

    props = {
        "colorbar": dict(
            tickmode="array",
            tickvals=decades,
            ticktext=np.round(10**decades, 2),
        ),
        **adds,
    }
    props.update(kwargs)

    hovertemplate = (
        f"<b>{trace.figure.layout.xaxis.title.text}</b>:"
        + "%{x}<br>"
        + f"<b>{trace.figure.layout.yaxis.title.text}</b>:"
        + "%{y}<br>"
        + f"<b>{trace.figure.layout.coloraxis.colorbar.title.text}</b>:"
        + "%{customdata:.2f}<extra></extra>"
    )
    trace.update(z=logz, hovertemplate=hovertemplate, customdata=10**logz)

    trace.figure.update_layout({trace.coloraxis: props})

    return trace.figure
