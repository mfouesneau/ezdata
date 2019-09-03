"""
Using Holoview and datashader to do some plots with Matplotlib

By default, holoview uses Bokeh. However, I personally do not see Bokeh capable
of publishable quality plots. This is probably because of my limited knowledge
of it.

As a result, I created this collection of plotting routines with holoview using
matplotlib instead. This approach is very similar to `hvplot`.

I defined my interfaces using the `ezdata.Plotter` API.


requirements
------------
datashader
"""

import pylab as plt
import numpy as np
# import datashader.transfer_functions as tf
import datashader
from .plotter import Plotter

# from holoviews.plotting.util import process_cmap
# Returns a palette


def get_doc_from(name, obj=plt):
    """ decorator to add documentation from a module (default: matplotlib)
    Parameters
    ----------
    name: str or callable
        name of the function to get the documentation from
        or function object
    obj: object
        module from which the function is an attribute
    Returns
    -------
    decorator: callable
        decorator
    """
    def deco(func):
        fn = getattr(name, '__doc__', None)
        if fn is None:
            fn = getattr(obj, name, None)
        if fn is not None:
            if func.__doc__ is None:
                func.__doc__ = fn.__doc__
            else:
                func.__doc__ += fn.__doc__
        return func
    return deco


def get_hv_canvas(dataframe, xname, yname,
                  x_range=None, y_range=None,
                  shape=256):
    """ Holoview Canvas definition common to many plots

        Parameters
        ----------
        dataframe: dictionary like object
            dataset
        xname: string
            Field name to draw x-positions from
        yname: string
            Field name to draw y-positions from
        shape: int or (int, int)
            shape of the density image

        Returns
        -------
        canvas : holoviews.Canvas
            canvas object
        extent: tuple
            xmin, xmax, ymin, ymax of the image
    """
    if x_range is None:
        xvalues = dataframe[xname]
        x_range = np.nanmin(xvalues), np.nanmax(xvalues)
    if y_range is None:
        yvalues = dataframe[yname]
        y_range = np.nanmin(yvalues), np.nanmax(yvalues)
    if np.ndim(shape) < 1:
        width = height = shape
    else:
        width, height = shape

    canvas = datashader.Canvas(plot_width=width, plot_height=height,
                               x_range=x_range, y_range=y_range)
    extent = x_range + y_range
    return canvas, extent


@get_doc_from(plt.imshow)
def _imshow_hv_map(agg, extent=None, **kwargs):
    """ Display image using imshow and some default values adapted to holoviews
    """
    alpha_mask = kwargs.pop('alpha_below', 0.)

    defaults = {'origin': 'lower',
                'aspect': 'auto',
                'extent': extent}

    defaults.update(**kwargs)

    return plt.imshow(np.ma.masked_less_equal(agg, alpha_mask), origin='lower',
                      extent=extent, aspect='auto', **kwargs)


def hv_scatter(dataframe, xname, yname,
               x_range=None, y_range=None, shape=256,
               # how='linear',
               what=datashader.count(), **kwargs):
    """ Scatter plot

        Parameters
        ----------
        xname: string
            Field name to draw x-positions from
        yname: string
            Field name to draw y-positions from
        **kwds : optional
            Keyword arguments to pass to `plt.imshow`

        Returns
        -------
        r : plt.ImageAxes
            result from `plt.imshow`
    """

    canvas, extent = get_hv_canvas(dataframe, xname, yname,
                                   x_range=x_range, y_range=y_range,
                                   shape=shape)

    try:
        agg = canvas.points(dataframe.to_pandas(), xname, yname, agg=what)
    except Exception:
        agg = canvas.points(dataframe, xname, yname, agg=what)

    # img = tf.shade(agg, how=how, cmap=cmap)
    # img.to_pil()

    return _imshow_hv_map(agg, extent=extent, **kwargs)


def hv_plot(dataframe, xname, yname,
            x_range=None, y_range=None, shape=256,
            # how='linear',
            what=datashader.count(), **kwargs):
    """ Line plot

        Parameters
        ----------
        xname: string
            Field name to draw x-positions from
        yname: string
            Field name to draw y-positions from
        **kwds : optional
            Keyword arguments to pass to `plt.imshow`

        Returns
        -------
        r : plt.ImageAxes
            result from `plt.imshow`
    """

    canvas, extent = get_hv_canvas(dataframe, xname, yname,
                                   x_range=x_range, y_range=y_range,
                                   shape=shape)

    try:
        agg = canvas.points(dataframe.to_pandas(), xname, yname, agg=what)
    except Exception:
        agg = canvas.points(dataframe, xname, yname, agg=what)

    return _imshow_hv_map(agg, extent=extent, **kwargs)


def hv_corner(df, varnames=None, shape=32, labels=None, figsize=None,
              lower_kwargs={}, diag_kwargs={}):
    """ Corner plot
    """
    if varnames is None:
        varnames = df.keys()
    if figsize is None:
        figsize = (3 * len(varnames), 3 * len(varnames))
    if labels is None:
        labels = varnames

    label_maps = {varname: label for varname, label in zip(varnames, labels)}

    plt.figure(figsize=figsize)
    pp = HvPlotter(df).pairplot(keys=varnames, labels=labels)
    kwargs = dict(only1d=True, bins=shape, edgecolor='k',
                  facecolor='None', histtype='step')
    kwargs.update(diag_kwargs)
    pp.map_diag('hist', **kwargs)
    plt.setp(plt.gcf().get_axes()[-1].get_xticklabels(), visible=True)
    kwargs = dict(cmap=plt.cm.hot, shape=shape)
    kwargs.update(lower_kwargs)
    pp.map_lower('scatter', **kwargs)
    # BUG
    plt.setp(pp.axes[-1][-1].get_xticklabels(), visible=True)
    corner_colorbar()

    # Quantiles
    for num, (kx, labelx) in enumerate(zip(varnames, labels)):
        ax = pp.axes[num][num]
        q_16, q_50, q_84 = np.quantile(df[kx], [0.16, 0.5, 0.84])
        q_m, q_p = q_50 - q_16, q_84 - q_50

        # Format the quantile display.
        fmt = "{{0:{0}}}".format(".2f").format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
        title = "{0} = {1}".format(labelx, title)
        ax.set_title(title, fontsize='medium')
        ylim = ax.get_ylim()
        ax.vlines([q_16, q_50, q_84], ylim[0], ylim[1],
                  color='k', linestyle='--')

    for axes in pp.axes:
        axes[0].set_ylabel(label_maps.get(axes[0].get_ylabel(), ''))
    for ax in pp.axes[-1][-len(varnames):]:
        ax.set_xlabel(label_maps.get(ax.get_xlabel(), ''))
    return pp


class HvPlotter(Plotter):
    """ A plotting wrapper around Holoview interfaces

    This should also work with pure dictionary objects.  all plotting functions
    are basically proxies to matplotlib in which arguments can be named columns
    from the data (not necessary) and each method handles a `ax` keyword to
    specify a Axes instance to use (default using :func:`plt.gca`)

    .. code-block:: python
        Plotter(df).groupby('class')\
                   .plot('RA', 'Dec', 'o', alpha=0.5, mec='None')

    Attributes
    ----------
    data: dict-like structure
        data with column named format
    label: str, optional
        label to use on the data as default label
    allow_expressions: bool, optional
        set to use math expressions with the keys
        see :func:`evalexpr`
    ax: plt.Axes instance
        contains the last axes reference(s) after a plot
        (do not exists if no plotting function was called)
    """
    def get(self, *args):
        from .simpletable import SimpleTable
        data = self.data
        try:
            df = {name: self._value_from_data(name) for name in args}
            return SimpleTable(df)
        except Exception:
            return data

    @get_doc_from(hv_scatter)
    def scatter(self, xname, yname, *args, **kwargs):
        """ Holoview Points wrapper """
        data = self.get(xname, yname)
        return hv_scatter(data, xname, yname, *args, **kwargs)

    @get_doc_from(hv_scatter)
    def plot(self, xname, yname, *args, **kwargs):
        """ Holoview Line wrapper """
        data = self.get(xname, yname)
        return hv_plot(data, xname, yname, *args, **kwargs)

    @get_doc_from(hv_corner)
    def corner(self, df, varnames=None, shape=32, labels=None,
               figsize=None, lower_kwargs={}, diag_kwargs={}):
        return hv_corner(df, varnames, shape, labels, figsize,
                         lower_kwargs, diag_kwargs)

    line = plot


def corner_colorbar(*args, **kwargs):
    """ Make a colorbar on a corner plot following the gaussian sigma
        prescription annotations """
    orientation = kwargs.get('orientation', 'vertical')
    ax = kwargs.get('ax', None)
    cax = kwargs.pop('cax', None)

    levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    if (ax is None) and (cax is None):
        if orientation == 'vertical':
            cax = plt.axes([0.8, 0.5, 0.03, 0.3])
        else:
            cax = plt.axes([0.6, 0.7, 0.3, 0.03])
    cb = plt.colorbar(cax=cax, **kwargs)
    cax = cb.ax
    cb.set_label('counts')
    vmin, vmax = cb.vmin, cb.vmax
    orientation = cb.orientation
    if orientation == 'vertical':
        xlim = cax.get_xlim()
        for k, val in enumerate(levels[1:][::-1], 1):
            denorm = (vmax - vmin) * val + vmin
            cax.hlines(denorm, xlim[0], xlim[1], 'w', lw=3)
            cax.hlines(denorm, xlim[0], xlim[1], 'k')
            cax.text(xlim[0], denorm, r'{0:d}$\,\sigma$    '.format(k),
                     ha='right', va='center')
    else:
        ylim = cax.get_ylim()
        for k, val in enumerate(levels[1:][::-1], 1):
            denorm = (vmax - vmin) * val + vmin
            cax.vlines(denorm, ylim[0], ylim[1], 'w', lw=3)
            cax.vlines(denorm, ylim[0], ylim[1], 'k')
            cax.text(denorm, ylim[1], r'{0:d}$\,\sigma$    '.format(k),
                     ha='center', va='bottom')
    return cb


class logcount(datashader.count):

    @staticmethod
    def _finalize(bases, **kwargs):
        return datashader.reductions.xr.DataArray(np.log10(bases[0]), **kwargs)
