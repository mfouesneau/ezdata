""" A convenient plotting container

In this package implements :class:`Plotter`, which is a simple container to
dictionary like structure (e.g. :class:`dict`, :class:`np.recarray`,
:class:`pandas.DataFrame`). It allows the user to plot directly using keys of
the data and also allows rapid group plotting routines (groupy and facets).

I was basically tired of all the packages doing fancy things and not allowing
basics or requiring a lot of dependencies.

Examples
--------

.. code-block::python

    >> d = {...}
    >> p = plotter.Plotter(d)
    >> g = p.groupby('BRK', markers='<^>v.oxs', colors='parula_r')
    >> g.plot('CRA', 'CDEC')
    >> g.colorbar().set_label('BRK')

Multiple groups can be done as well. (Caution, the `facet` option is not
robust)

.. code-block::python

    >> g = p.groupby('BRK', facet=True, sharex=True, sharey=True)\
            .groupby('FLD')
    >> g.plot('CRA', 'CDEC', 'o')

.. note::

    * tested with python 2.7, & 3.4
    * tested compatible with pandas (not required)
    * requirements: numpy, matplotlib

:author: Morgan Fouesneau
"""

from __future__ import (absolute_import, division, print_function)

import sys
import pylab as plt
import matplotlib as mpl
import numpy as np
import itertools
from matplotlib.ticker import MaxNLocator

from . import astro

PY3 = sys.version_info[0] > 2

if PY3:
    basestring = (str, bytes)
else:
    basestring = (str, unicode)


__all__ = ['Group', 'Plotter', 'create_common_cbar', 'colorify',
           'evalexpr', 'create_common_legend']


def get_doc_from(name, obj=plt):
    """ decorator to add documentation from a module (default: matplotlib)

    Parameters
    ----------
    name: str
        name of the function to get the documentation from
    obj: object
        module from which the function is an attribute

    Returns
    -------
    decorator: callable
        decorator
    """
    def deco(func):
        fn = getattr(obj, name, None)
        if fn is not None:
            if func.__doc__ is None:
                func.__doc__ = fn.__doc__
            else:
                func.__doc__ += fn.__doc__
        return func
    return deco


def _arg_groupby(data, key):
    """ create an iterator which returns (key, index) grouped by each
    value of key(value) """
    val = data[key]

    def parse_missing_data(x, dtype=str):
        """ Make sure null/missing values are still sorted """
        cond = x in [None, '', 'None', 'none',
                     float('nan'), 'nan', 'NaN',
                     'null', 'Null',
                     float('inf'), 'inf']
        return cond, x

    ind = sorted(zip(val, range(len(val))), key=lambda x: parse_missing_data(x[0]))

    for k, grp in itertools.groupby(ind, lambda x: x[0]):
        index = [k[1] for k in grp]
        yield k, index


def _groupby(data, key):
    """ create an iterator which returns (key, DataFrame) grouped by each
    value of key(value) """
    for k, index in _arg_groupby(data, key):
        d = {a: b[index] for a, b in data.items()}
        yield k, data.__class__(d)


def _safe_compute(val):
    """ Return the computed value if necessary

    Dask DataFrame may sometimes need to call compute to interact
    properly with some calls. This method if a shortcut to make sure
    we work with values when necessary.
    """
    try:
        return val.compute()
    except AttributeError:
        return val


class Group(object):
    """ Group multiple plotter instances into one container. This offers any
    function of :class:`Plotter` through an implicit loop of any method It
    allows for instance to generate multiple plots on the same axes or even
    facet plot (one per group).

    .. code-block:: python

        >> g = Plotter(df).groupby('class')
        >> g.set_options(facet=True, ncols=2, projection='aitoff')
        # which is equivalent to
        >> g = Plotter(df)\
                .groupby('class', facet=True, ncols=2, projection='aitoff')
        >> g.plot('RA', 'Dec', 'o', alpha=0.5, mec='None')

    Attributes
    ----------
    seq: sequence
        Sequence of Plotter instances
    title: str
        name of the group (used as label is nested groups)
    facet: bool
        set to use facets, i.e., one subplot per element of the group
    markers: iterable
        sequence of markers one per group
    linestyles: iterable
        sequence of linestyles one per group
    colors: seq or Colormap
        sequence of colors or Colormap instance from which deriving a
        sequence of colors to encode each group
        if Colormap instance, a cmap attribute will be generated after a
        plot and will refer to the updated instance
    sharex: bool
        set to share x-axis with all subplots
    sharey: bool
        set to share y-axis with all subplots
    kwargs: dict
        any other option will be forwarded to :func:`plt.subplot`

    .. see also::

        :func:`set_options`
    """
    def __init__(self, seq, title='', **kwargs):
        self.seq = seq
        self.title = title
        self.facet = False
        self.markers = None
        self.linestyles = None
        self.colors = None
        self.ncols = 3
        self.sharex = False
        self.sharey = False
        self.axes = None
        self.kwargs = {}
        self._all_against = False
        self._show_facet_titles = True
        self.create_common_cbar = create_common_cbar
        self.set_options(**kwargs)
        self.show = plt.show

    def make_facets(self, show_titles=True):
        """ generates multiple subplots
        uses self.ncols as number of columns
        and subplots are also using self.kwargs.

        Returns
        -------
        axes: sequence
            sequence of the axes instance from the subplots

        show_titles: bool
            set to add the title of the subplot to the group's name

        .. see also::

            :func:`set_options`
        """
        axes = []
        n = len(self)
        ncols = min(self.ncols, n)
        nlines = n // ncols
        if ncols * nlines < n:
            nlines += 1
        if nlines == 0:
            nlines = 1
            ncols = n
        axes = []
        ax = sharex = sharey = None
        for k in range(n):
            if self.sharex:
                sharex = ax
            if self.sharey:
                sharey = ax
            ax = plt.subplot(nlines, ncols, k + 1, sharex=sharex,
                             sharey=sharey, **self.kwargs)
            axes.append(ax)
            if (self.seq[k].label is not None) and self._show_facet_title:
                ax.set_title(self.seq[k].label)
            if (self.sharex):
                if k < (n - ncols):
                    plt.setp(ax.get_xticklabels(), visible=False)
            if (self.sharey):
                if (k % ncols) > 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
        self.axes = axes
        return axes

    def set_options(self, **kwargs):
        """ Set some options

        Parameters
        ----------
        title: str
            rename the group
        facet: bool
            set the group to display facets or one plot
        ncols: int
            when facet is True, this gives how many columns should be used
        markers: seq
            sequence of markers (will cycle through)
        linestyles: seq
            sequence of linestyles (will cycle through)
        colors: seq or Colormap
            sequence of colors or Colormap instance from which deriving a
            sequence of colors to encode each group
            if Colormap instance, a cmap attribute will be generated after a
            plot and will refer to the updated instance
        labels: seq
            Labels used for each group instead of values
        sharex: bool
            set to share x-axis with all subplots
        sharey: bool
            set to share y-axis with all subplots
        all_against: bool
            set if plotting variables against a common one
        kwargs: dict
            any other option will be forwarded to :func:`plt.subplot`

        Returns
        -------
        self: Group instance
            returns itself for conveniance when writting one liners.
        """
        title = kwargs.pop('title', None)
        facet = kwargs.pop('facet', None)
        ncols = kwargs.pop('ncols', None)
        markers = kwargs.pop('markers', None)
        colors = kwargs.pop('colors', None)
        linestyles = kwargs.pop('linestyles', None)
        labels = kwargs.pop('labels', None)
        sharex = kwargs.pop('sharex', None)
        sharey = kwargs.pop('sharey', None)
        allow_expressions = kwargs.pop('allow_expressions', None)
        self._all_against = kwargs.pop('all_against', self._all_against)
        self._show_facet_title = kwargs.pop('show_facet_title',
                                            self._show_facet_title)
        self.ncols = kwargs.pop('ncols', self.ncols)
        if sharex is not None:
            self.sharex = sharex
        if sharey is not None:
            self.sharey = sharey
        if title is not None:
            self.title = title
        if facet is not None:
            self.facet = facet
        if ncols is not None:
            self.ncols = ncols
        if markers is not None:
            self.markers = markers
        if colors is not None:
            self.colors = colors
            if isinstance(self.colors, basestring):
                self.colors = plt.cm.get_cmap(self.colors)
        if linestyles is not None:
            self.linestyles = linestyles
        if labels is not None:
            for k, v in zip(self.seq, itertools.cycle(labels)):
                k.label = v
        if allow_expressions is not None:
            for k in self.seq:
                k.allow_expressions = allow_expressions
        self.kwargs.update(kwargs)
        return self

    def groupby(self, key, select=None, labels=None, **kwargs):
        """ Make individual plots per group

        Parameters
        ----------
        key: str
            key on which building groups

        select: sequence
            explicit selection on the groups
            if a group does not exist, it will be returned empty

        labels: dict
            set to replace the group names by a specific label string during
            the plot

        kwargs: dict
            optional keywords forwarded to :func:`set_options` method

        Returns
        -------
        g: Group instance
            group of plotters

        .. see also::

            :func:`set_options`
        """
        gg = []
        for sk in self.seq:
            lst = sk.groupby(key, select=select, labels=labels)
            for k, v in sk.__dict__.items():
                if k not in ['seq', 'title']:
                    setattr(lst, k, v)
                if getattr(sk, 'title', None) is not None:
                    lst.label = sk.title
            lst.set_options(**kwargs)
            gg.append(lst)
        return self.__class__(gg, title=self.title)

    def subplot(self, *args, **kwargs):
        """ A convenient shortcut for one liner use
        Generates a subplot with given arguments and returns `self`.
        """
        self.axes = plt.subplot(*args, **kwargs)
        return self

    def apply(self, fn, *args, **kwargs):
        """apply function on each element of the group

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.
        """
        if self.facet:
            axes = self.make_facets()
        ret = []
        for ax, element in zip(axes, self.seq):
            plt.sca(ax)
            ret.append(element.apply(fn, *args, **kwargs))
        return ret

    def __len__(self):
        return len(self.seq)

    def __repr__(self):
        txt = """Object Group {0:s} (length={2:d}): {1:s}"""
        return txt.format(self.title, object.__repr__(self), len(self))

    def __dir__(self):
        """ show the content of Plotter """
        return self.seq[0].__dir__()

    def __getattr__(self, k):
        """ Returns a looper function on each plotter of the group """
        cyclenames = 'linestyles', 'colors', 'markers'
        cyclekw = {k: getattr(self, k) for k in cyclenames}
        if isinstance(self.colors, mpl.colors.Colormap):
            if not self._all_against:
                s = set()
                for sk in self.seq:
                    s = s.union(set(sk.data[self.title]))
            else:
                s = np.arange(len(self.seq))
            colors, cmap = colorify(s)
            cyclekw['colors'] = colors
            self.cmap = cmap
        elif self.colors is None:
            cyclekw['colors'] = plt.rcParams['axes.prop_cycle']\
                .by_key()['color']
        cc_ = mpl.colors.ColorConverter()
        cyclekw['colors'] = [cc_.to_rgba(val) for val in cyclekw['colors']]
        if self.facet:
            axes = self.make_facets()
            return self.looper_facet_method(self.seq, k, axes, cyclekw=cyclekw)
        else:
            return self.looper_method(self.seq, k, cyclekw=cyclekw)

    def __iter__(self):
        """ Iterator over the individual plotter of the group """
        for k in self.seq:
            yield k

    def __getitem__(self, k):
        """ Returns one plotter of the group """
        return self.seq[k]

    @staticmethod
    def looper_method(lst, methodname, cyclekw={}, **kw):
        """ calls a method on many instance of sequence of objects

        Parameters
        ----------
        lst: sequence
            sequence of objects to call the method from
        methodname: str
            name of the method to call from each object
        cyclekw: dict
            keyword arguments that calls need to cycle over per object.
            Each element in this dictionary is expected to be a sequence and
            one element of each will be used per call. It will use
            :func:`itertools.cycle`. (None elements are filtered)
            cyclenames = 'linestyles', 'colors', 'markers'
        kw: dict
            other keywords (have priority on `cyclekw`)

        Returns
        -------
        deco: callable
            mapper function
        """

        cyclenames = 'linestyles', 'colors', 'markers'

        _cyclekw = {k: itertools.cycle(cyclekw[k])
                    for k in cyclenames if cyclekw[k] is not None}

        def next_cyclekw():
            a = {k[:-1]: next(v) for k, v in _cyclekw.items()}
            return a

        def deco(*args, **kwargs):
            r = []
            for l in lst:
                k0 = next_cyclekw()
                kw.update(k0)
                kw.update(kwargs)
                if (l.data is None) or _safe_compute(np.size(l.data) == 0):
                    a = None
                else:
                    a = getattr(l, methodname)(*args, **kw)
                r.append(a)
            return r
        return deco

    @staticmethod
    def looper_facet_method(lst, methodname, axes, cyclekw={}, **kw):
        """
        calls a method on many instance of sequence of objects but also imposes
        ax as keyword argument. This method will also test if there is no data
        to plot.

        Parameters
        ----------
        lst: sequence
            sequence of objects to call the method from
        methodname: str
            name of the method to call from each object
        axes: sequence
            list of axes, one per call
        cyclekw: dict
            keyword arguments that calls need to cycle over per object.  Each
            element in this dictionary is expected to be a sequence and one
            element of each will be used per call. It will use
            :func:`itertools.cycle`. (None elements are filtered)
            cyclenames = 'linestyles', 'colors', 'markers'
        kw: dict
            other keywords (have priority on `cyclekw`)

        Returns
        -------
        deco: callable
            mapper function
        """
        cyclenames = 'linestyles', 'colors', 'markers'

        _cyclekw = {k: itertools.cycle(cyclekw[k])
                    for k in cyclenames if cyclekw[k] is not None}

        def next_cyclekw():
            a = {k[:-1]: next(v) for k, v in _cyclekw.items()}
            return a

        def deco(*args, **kwargs):
            r = []
            for l, ax in zip(lst, axes):
                k0 = next_cyclekw()
                kw.update(k0)
                kw.update(kwargs)
                if (l.data is None) or (np.size(l.data) == 0):
                    _intercept_empty_plot(ax=ax)
                else:
                    kw['ax'] = ax
                    a = getattr(l, methodname)(*args, **kw)
                    r.append(a)
            return r
        return deco

    @get_doc_from('colorbar')
    def colorbar(self, *args, **kwargs):
        if not hasattr(self, 'cmap'):
            print('No registered colormap with the group')
            return
        return plt.colorbar(self.cmap, *args, **kwargs)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Group([self.seq, other])
        elif isinstance(other, self[0].__class__):
            # copy the list and append the new element
            g = self.__class__([k for k in self.seq])
            g.seq.append(other)
            for k, v in self.__dict__.items():
                if k not in ['seq', 'title']:
                    setattr(g, k, v)
            return g
        else:
            raise RuntimeError('Cannot add {0} type objects to {1} instance'
                               .format(other.__class__.__name__,
                                       self.__class__.__name__))

    def pairplot(self, keys=None, **kwargs):
        """ This is a high-level interface for PairGrid
        intended to make it easy to draw a few common styles.

        Parameters
        ----------
        keys: sequence
            work with only a set of keys if provided.

        Returns
        -------
        p: PairGrid instance
            :class`PairGrid` which gives you all possible flexibility.
        """
        if keys is None:
            keys = self.data.keys()
        return PairGrid(self, keys, allow_expressions=self.allow_expressions,
                        **kwargs)


class Plotter(object):
    """
    A wrapper around plotting functions and DataFrame
    This should also work with pure dictionary objects.

    all plotting functions are basically proxies to matplotlib in which
    arguments can be named columns from the data (not necessary) and each
    method handles a `ax` keyword to specify a Axes instance to use (default
    using :func:`plt.gca`)

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
    def __init__(self, data, label=None, allow_expressions=False,
                 update_axis_label=True):
        self.data = data
        self.label = label
        self.allow_expressions = allow_expressions
        self.show = plt.show
        self.update_axis_label = update_axis_label
        self.label = label

    def set_options(self, **kwargs):
        self.label = kwargs.get('label', self.label)
        self.allow_expressions = kwargs.get('allow_expressions',
                                            self.allow_expressions)
        self.show = kwargs.get('show', self.show)
        self.update_axis_label = kwargs.get('update_axis_label', self.label)
        return self

    def _ensure_data_type(self, data):
        """ Make sure data is compatible with a dictionary like interface """
        if isinstance(data, dict) or hasattr(data, '__getitem__'):
            return data

        # assuming array
        data = {e: k for e, k in enumerate(np.asarray(data).T)}
        return data

    @property
    def keys(self):
        try:
            return self.data.keys()
        except AttributeError:
            # Dataframes from pandas and dask
            return self.data.columns

    def _value_from_data(self, key):
        """ Parse a key for existing data in the dataframe. If not found,
        returns the key directly """
        if not isinstance(key, basestring):
            value = key
        elif key not in self.data:
            if self.allow_expressions:
                try:
                    value = evalexpr(self.data, key)
                except Exception:
                    pass
            value = key
        else:
            value = self.data[key]

            return _safe_compute(value)

    def _select_data(self, selection):
        """ Parse indices or expression and return selected data"""
        try:
            return self.data.selectWhere('*', selection)
        except ValueError:
            return self.data.select('*', indices=np.where(selection)[0])
        except AttributeError or KeyError:
            if selection is None:
                return self.data
            elif isinstance(selection, basestring):
                indexes = self.data.eval(selection)
                return self.data.where(indexes)
            else:
                return self.data[selection]

    def select(self, selection, labels=None, **kwargs):
        """ Returns a Group from selected data

        Parameters
        ----------
        selection: str or expression or sequence of these
            the selection could be a sequence of selections
            A given selection on the data can be an string or the evaluation of
            it (boolean array or dask array)

        labels: sequence(str)
            the labels of the selections used in the Group

        Returns
        -------
        group: Group instance
            group of plotters (one per selection)
        """
        if isinstance(selection, (basestring, type(None))):
            selection = [selection]

        if labels is None:
            labels = []
            for num, select in enumerate(selection, 1):
                if isinstance(select, (basestring, type(None))):
                    labels.append(str(select))
                else:
                    labels.append('subset {0:d}'.format(num))

        elements = []
        for select, label in zip(selection, labels):
            subdata = self._select_data(select)
            elements.append(self.__class__(subdata, label=label))

        return Group(elements).set_options(**kwargs)

    def evalexpr(self, expr, exprvars=None, dtype=float):
        """ evaluate expression based on the data and external variables
            all np function can be used (log, exp, pi...)

        Parameters
        ----------
        data: dict or dict-like structure
            data frame / dict-like structure containing named columns

        expr: str
            expression to evaluate on the table
            includes mathematical operations and attribute names

        exprvars: dictionary, optional
            A dictionary that replaces the local operands in current frame.

        dtype: dtype definition
            dtype of the output array

        Returns
        -------
        out : np.array
            array of the result
        """
        return evalexpr(self.data, expr, exprvars=exprvars, dtype=dtype)

    def subplot(self, *args, **kwargs):
        """ A convenient shortcut for one liner use
        Generates a subplot with given arguments and returns self.
        """
        plt.subplot(*args, **kwargs)
        return self

    def colorify(self, key, vmin=None, vmax=None, cmap=None):
        """ Associate a color map to a quantity vector

        Parameters
        ----------
        data: sequence
            values to encode
        vmin: float
            minimum value
        vmax: float
            maximum value
        cmap: Colormap instance
            colormap to use

        Returns
        -------
        colors: sequence or array
            one color per input data
        cmap: Colormap
            data normalized colormap instance
        """
        return colorify(self.data.evalexpr(key), vmin, vmax, cmap)

    def apply(self, fn, *args, **kwargs):
        """ Apply an arbitrary function on the data to plot
        The first argument of that function must be the dataset
        By default uses the active axes.

        Parameters
        ----------
        fn: callable
            plotting function to apply with args and kwargs arguments

        Returns
        -------
        r: tuple
            anything that fn returns.
        """
        if isinstance(fn, basestring):
            _fn = getattr(self, fn)
            if _fn is None:
                raise AttributeError('No function named {0:s}'.format(fn))
            return _fn(*args, **kwargs)
        else:
            return fn(self.data, *args, **kwargs)

    @get_doc_from('xlabel')
    def xlabel(self, *args, **kwargs):
        """ Set the xlabel of the current plot """
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        ax.set_xlabel(*args, **kwargs)
        return self

    @get_doc_from('ylabel')
    def ylabel(self, *args, **kwargs):
        """ Set the ylabel of the current plot """
        ax = kwargs.pop('ax', None)
        if ax is None:
            plt.gca()
        ax.set_ylabel(*args, **kwargs)
        return self

    def _set_auto_axis_labels(self, xname, yname, ax=None):
        """ convinient shortcut for labelling axis """
        if not self.update_axis_label:
            return
        if ax is None:
            ax = plt.gca()
        if xname is not None:
            ax.set_xlabel(xname)
        if yname is not None:
            ax.set_ylabel(yname)

    @get_doc_from('scatter')
    def scatter(self, x, y, c='k', s=20, *args, **kwargs):
        _x = self._value_from_data(x)
        _y = self._value_from_data(y)
        c = kwargs.pop('color', c)
        _c = np.atleast_2d(self._value_from_data(c))
        s = kwargs.pop('size', s)
        _s = np.atleast_2d(self._value_from_data(s))
        ax = kwargs.pop('ax', None)

        if ax is None:
            ax = plt.gca()

        if 'label' not in kwargs:
            kwargs['label'] = self.label

        self._set_auto_axis_labels(x, y, ax=ax)

        return ax.scatter(_x, _y, c=_c, s=_s, *args, **kwargs)

    @get_doc_from('plot')
    def plot(self, x, y, *args, **kwargs):
        _x = self._value_from_data(x)
        _y = self._value_from_data(y)
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        self.axes = ax

        if 'label' not in kwargs:
            kwargs['label'] = self.label

        self._set_auto_axis_labels(x, y, ax=ax)

        return ax.plot(_x, _y, *args, **kwargs)

    @get_doc_from('bar')
    def bar(self, x, y, *args, **kwargs):
        _x = self._value_from_data(x)
        _y = self._value_from_data(y)
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        self.axes = ax

        if 'label' not in kwargs:
            kwargs['label'] = self.label

        self._set_auto_axis_labels(x, y, ax=ax)

        return ax.bar(_x, _y, *args, **kwargs)

    @get_doc_from('step')
    def step(self, x, y, *args, **kwargs):
        _x = self._value_from_data(x)
        _y = self._value_from_data(y)
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        self.axes = ax

        if 'label' not in kwargs:
            kwargs['label'] = self.label

        self._set_auto_axis_labels(x, y, ax=ax)

        return ax.step(_x, _y, *args, **kwargs)

    @get_doc_from('hist')
    def hist(self, x, *args, **kwargs):
        _x = self._value_from_data(x)
        ind = np.isfinite(_x)
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        self.axes = ax

        if 'label' not in kwargs:
            kwargs['label'] = str(self.label)

        self._set_auto_axis_labels(x, None, ax=ax)

        ind = np.isfinite(_x)
        _w = kwargs.pop('weights', None)
        if _w is not None:
            return ax.hist(_x[ind], weights=_w[ind], *args, **kwargs)
        else:
            return ax.hist(_x[ind], *args, **kwargs)

    @get_doc_from('hist2d')
    def hist2d(self, x, y, *args, **kwargs):
        _x = self._value_from_data(x)
        _y = self._value_from_data(y)
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        self.axes = ax

        if 'label' not in kwargs:
            kwargs['label'] = self.label

        ind = np.isfinite(_x) & np.isfinite(_y)
        _w = kwargs.pop('weights', None)

        self._set_auto_axis_labels(x, y, ax=ax)

        if _w is not None:
            return ax.hist2d(_x[ind], _y[ind], weights=_w[ind],
                             *args, **kwargs)
        else:
            return ax.hist2d(_x[ind], _y[ind], *args, **kwargs)

    @get_doc_from('hexbin')
    def hexbin(self, x, y, C=None, *args, **kwargs):
        _x = self._value_from_data(x)
        _y = self._value_from_data(y)
        _C = self._value_from_data(C)
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        self.axes = ax

        if 'label' not in kwargs:
            kwargs['label'] = self.label

        self._set_auto_axis_labels(x, y, ax=ax)

        if _C is not None:
            ind = np.isfinite(_x) & np.isfinite(_y) & np.isfinite(_C)
            return ax.hexbin(_x[ind], _y[ind], C=_C[ind], *args, **kwargs)
        else:
            ind = np.isfinite(_x) & np.isfinite(_y)
            return ax.hexbin(_x[ind], _y[ind], *args, **kwargs)

    @get_doc_from('violinplot')
    def violinplot(self, dataset, **kwargs):
        d = (self._value_from_data(k) for k in dataset)
        d = [dk[np.isfinite(dk)] for dk in d]
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        self.axes = ax
        if 'label' not in kwargs:
            kwargs['labels'] = dataset
        return ax.violinplot(d, **kwargs)

    @get_doc_from('boxplot')
    def boxplot(self, dataset, **kwargs):
        d = [self._value_from_data(k) for k in dataset]
        d = [dk[np.isfinite(dk)] for dk in d]
        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        self.axes = ax
        if 'labels' not in kwargs:
            kwargs['labels'] = dataset
        return ax.boxplot(d, **kwargs)

    @get_doc_from('healpix_plot', astro)
    def healpix_plot(self, *args, **kwargs):
        return astro.healpix_plot(self.data, *args, **kwargs)

    @get_doc_from('project_aitoff', astro)
    def plot_aitoff(self, alpha, delta,
                    radians=False, **kwargs):
        """plot aitoff projection
        (https://en.wikipedia.org/wiki/Aitoff_projection) projection

        Parameters
        ----------
        alpha: array
            azimuth angle
        delta: array
            polar angle
        radians: boolean
            input and output in radians (True), or degrees (False)
        """
        x, y = astro.project_aitoff(self.data[alpha],
                                    self.data[delta],
                                    radians=radians)
        return self.__class__(dict(aitoff_x=x, aitoff_y=y))\
                   .plot('x', 'y', **kwargs)

    def groupby(self, key, select=None, labels=None, **kwargs):
        """ Make individual plots per group

        Parameters
        ----------
        key: str
            key on which building groups

        select: sequence
            explicit selection on the groups
            if a group does not exist, it will be returned empty

        labels: dict
            set to replace the group names by a specific label string during
            the plot

        Returns
        -------
        g: Group instance
            group of plotters
        """
        r = _groupby(self.data, key)

        if select is not None:
            grp = dict((k, v) for k, v in r if k in select)
            r = [(k, grp.get(k, [])) for k in select]

        if labels is None:
            labels = {}

        lst = [self.__class__(g, label=labels.get(k, k)) for k, g in r]
        return Group(lst, title=key).set_options(**kwargs)

    def all_against(self, key, select=None, labels=None, **kwargs):
        """ Make individual plots per of all variables against one

        Parameters
        ----------
        key: str
            key on which plotting everything

        select: sequence
            explicit selection on the groups
            if a group does not exist, it will be returned empty

        labels: dict
            set to replace the names by a specific label string during
            the plot

        Returns
        -------
        g: Group instance
            group of plotters
        """
        r = ((other, {key: self.data[key], 'value': self.data[other]})
             for other in self.data.keys() if other != key)

        if select is not None:
            grp = dict((k, v) for k, v in r if k in select)
            r = [(k, grp.get(k, [])) for k in select]

        if labels is None:
            labels = {}

        lst = [self.__class__(g, label=labels.get(k, k)) for k, g in r]
        return Group(lst, title=key, all_against=True).set_options(**kwargs)

    def lagplot(self, x, t=1, **kwargs):
        """
        A lag plot checks whether a data set or time series is random or not.

        Random data should not exhibit any identifiable structure in the lag
        plot. Non-random structure in the lag plot indicates that the
        underlying data are not random.

        Parameters
        ----------
        x: str
            the data column to plot
        t: int
            lag to apply, default 1

        see also: :func:`scatter`
        """
        _x = self._value_from_data(x)
        _y = np.hstack([_x[t:], _x[:t]])

        ax = kwargs.pop('ax', None)
        if ax is None:
            ax = plt.gca()
        self.axes = ax

        if 'label' not in kwargs:
            kwargs['label'] = self.label

        defaults = dict(marker='o', linestyle='None')
        defaults.update(kwargs)

        return ax.plot(_x, _y, **defaults)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return Group([self, other])
        else:
            raise RuntimeError('Cannot add {0} type objects to {1} instance'
                               .format(other.__class__.__name__,
                                       self.__class__.__name__))

    def pivot_plot(self, key1, key2, plotfn, plotkw={}, **kwargs):
        """ generate a multiple plots ordered according to 2 keys

        Parameters
        ----------
        key1: str
            key along the x-axis
        key2: str
            key along the y-axis
        plotfn: callable
            the plotting function
            This function signature must take a dataset and manage an `ax`
            keyword
            > plotfn(data, ax=ax, **plotkw)
        plotkw: dict
            optional keywords to pass to the plotting function
        kwargs: dict
            forwarded to :func:`plt.subplots`

        Returns
        -------
        axes: sequence
            list of all axes used in the plot
        """

        grp = self.aggregate(lambda x: x, (key1, key2))

        sx = {k[0] for k in grp}
        sx = {k: e for e, k in enumerate(sx)}
        sy = {k[1] for k in grp}
        sy = {k: e for e, k in enumerate(sy)}

        defaults = dict(sharex=True, sharey=True)
        defaults.update(**kwargs)

        fig, axes = plt.subplots(len(sy), len(sx), **defaults)
        _axes = np.rot90(axes, 3)
        for (idx1, idx2, data) in grp:
            e1, e2 = sx[idx1], sy[idx2]
            plotfn(data, ax=_axes[e1, e2], **plotkw)
            _axes[e1, e2].set_xlabel('')
            _axes[e1, e2].set_ylabel('')

        for ax in axes.ravel():
            plt.setp(ax.get_yticklines() + ax.get_xticklines(),
                     visible=False)
            plt.setp(ax.get_yticklabels() + ax.get_xticklabels(),
                     visible=False)

        return axes

    def aggregate(self, func, keys, args=(), kwargs={}):
        """ Apply func on groups within the data

        Parameters
        ----------
        func: callable
            function to apply
        keys: sequence(str)
            sequence of keys defining the groups
        args: tuple
            optional arguments to func (will be given at the end)
        kwargs: dict
            optional keywords to func

        Returns
        -------
        seq: sequence
            flattened sequence of keys and value
            (key1, key2, ... keyn, {})
        """
        pv = [(k, list(v)) for k, v in self.multigroupby(self.data, *keys)]

        def _aggregate(a, b=()):
            data = []
            for k, v in a:
                if type(v) in (list, tuple,):
                    data.append(_aggregate(v, b=(k,)))
                else:
                    data.append(b + (k, func(v)))
            return data

        return list(itertools.chain(*_aggregate(pv)))

    def multigroupby(self, *args):
        """
        Generate nested df based on multiple grouping keys

        Parameters
        ----------
        args: str or sequence
            column(s) to index the DF
        """
        if len(args) <= 0:
            yield self.data
        elif len(args) > 1:
            nested = True
        else:
            nested = False

        val = self.data[args[0]]
        ind = sorted(zip(val, range(len(val))), key=lambda x: x[0])

        for k, grp in itertools.groupby(ind, lambda x: x[0]):
            index = [v[1] for v in grp]
            d = self.data.ary.__class__({a: np.array([b[i] for i in index]) for
                                         a, b in self.data.items()})
            if nested:
                yield k, self.multigroupby(d, *args[1:])
            else:
                yield k, d

    def pairplot(self, keys=None, **kwargs):
        """ This is a high-level interface for PairGrid
        intended to make it easy to draw a few common styles.

        Parameters
        ----------
        keys: sequence
            work with only a set of keys if provided.

        Returns
        -------
        p: PairGrid instance
            :class`PairGrid` which gives you all possible flexibility.
        """
        if keys is None:
            keys = self.data.keys()
        return PairGrid(self, keys, allow_expressions=self.allow_expressions,
                        **kwargs)

    def cornerplot(self, varnames=None, labels=None, figsize=None, **kwargs):
        """ This is a high-level interface for PairGrid making quickly a CornerPlot

        Parameters
        ----------
        plotter: Plotter instance
            plotter to use. If a dataframe is provided, the default will be to
            use Plotter(plotter).

        varnames: seq(str)
            limit the plot to a subset of variables

        labels: seq(str)
            replace the variable names by provided labels

        figsize: tuple(height, width)
            Size of the figure. Default is a square of length 3 * len(varnames)

        kwargs: dict
            Forwarded to `Plotter.pairplot` method
        """
        return CornerPlot(self, varnames=varnames, labels=labels,
                          figsize=figsize, **kwargs)


class PairGrid(object):
    """ Container to Plot pairwise relationships in a dataset.

    By default, this function will create a grid of Axes such that each
    variable in data will by shared in the y-axis across a single row and in
    the x-axis across a single column. The diagonal Axes could be treated
    differently: for instance drawing a plot to show the univariate
    distribution of the data for the variable in that column.

    It is also possible to show a subset of variables or plot different
    variables on the rows and columns. This class works also with Group
    instances.
    """

    def __init__(self, data, keys, allow_expressions=False, **kwargs):
        self.data = data
        self.keys = list(keys)
        self.allow_expressions = allow_expressions
        self.show = plt.show
        nlines = ncols = len(self.keys)
        self.shape = (nlines, ncols)
        self.axes = np.empty((nlines, ncols), dtype=object)
        self.axes_dims = []
        self.set_options(**kwargs)
        self._generate_grid()

    def set_options(self, **kwargs):
        self.lbls = kwargs.pop('labels', self.keys)
        self.ticksrotation = kwargs.pop('ticksrotation', 0)
        self.weights = kwargs.pop('weights', None)
        self.max_n_ticks = kwargs.pop('max_n_ticks', 5)
        return self

    def adjust(self, left=None, bottom=None, right=None, top=None,
               wspace=None, hspace=None):
        """ adjust spacing between subplots using mpl.subplot_adjusts. """
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        return self

    def _check_label_visibility(self):

        upper_visible = self.axes[0][-1]._visible
        lower_visible = self.axes[-1][0]._visible
        diago_visible = self.axes[0][0]._visible
        n_axes = len(self.axes)

        if diago_visible:
            # check the diagonal labels
            for k in range(n_axes):
                ax = self.axes[k][k]
                plt.setp(ax.get_xticklabels(), rotation=self.ticksrotation,
                         visible=not lower_visible or (k == n_axes-1))
                plt.setp(ax.get_yticklabels(),
                         visible=not (upper_visible and lower_visible))
                ax.tick_params(top=False,
                               left=not lower_visible,
                               right=(not upper_visible) and lower_visible,
                               labelleft=not lower_visible,
                               labelright=(not upper_visible
                                           and lower_visible))
                ax.spines['right'].set_visible(lower_visible
                                               and not upper_visible)
                ax.spines['left'].set_visible(not lower_visible)
                ax.spines['top'].set_visible(False)
                if lower_visible:
                    ax.set_xlabel('')
            ax = self.axes[-1][-1]
            ax.set_xlabel(self.lbls[-1])
            plt.setp(ax.get_xticklabels(), visible=True)

        if upper_visible:
            for i in range(n_axes):
                for j in range(i + 1, n_axes):
                    ax = self.axes[i][j]
                    ax.spines['right'].set_visible(True)
                    ax.spines['left'].set_visible(False)
                    ax.spines['top'].set_visible(True)
                    ax.spines['bottom'].set_visible(False)
                    ax.tick_params(top=True,
                                   bottom=False,
                                   left=False,
                                   right=True,
                                   labelleft=False,
                                   labelright=True,
                                   labelbottom=False,
                                   labeltop=True)
                    plt.setp(ax.get_xticklabels(), rotation=self.ticksrotation,
                             visible=False or (i == 0))
                    plt.setp(ax.get_yticklabels(), rotation=self.ticksrotation,
                             visible=False or (j == n_axes -1))
                    ax.xaxis.set_label_position('top')
                    ax.yaxis.set_label_position('right')
                    ax.set_xlabel('')
                    ax.set_ylabel('')

            for k in range(1, n_axes):
                ax = self.axes[0][k]
                ax.set_xlabel(self.lbls[k])
                plt.setp(ax.get_xticklabels(), visible=True)

            for k in range(0, n_axes - 1):
                ax = self.axes[k][-1]
                ax.set_ylabel(self.lbls[k])
                plt.setp(ax.get_yticklabels(), visible=True)

        if lower_visible:
            for i in range(n_axes):
                for j in range(0, i):
                    ax = self.axes[i][j]
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(True)
                    ax.spines['top'].set_visible(False)
                    ax.spines['bottom'].set_visible(True)
                    ax.tick_params(top=False,
                                   bottom=True,
                                   right=False,
                                   left=True)
                    plt.setp(ax.get_xticklabels(), rotation=self.ticksrotation,
                             visible=False or (i == n_axes -1))
                    plt.setp(ax.get_yticklabels(), rotation=self.ticksrotation,
                             visible=False or (j == 0))
                    ax.set_xlabel('')
                    ax.set_ylabel('')

            for k in range(0, n_axes - 1):
                ax = self.axes[-1][k]
                ax.set_xlabel(self.lbls[k])
                plt.setp(ax.get_xticklabels(), visible=True)

            for k in range(1, n_axes):
                ax = self.axes[k][0]
                ax.set_ylabel(self.lbls[k])
                plt.setp(ax.get_yticklabels(), visible=True)

    def _generate_grid(self):

        nlines, ncols = self.shape
        keys = self.keys

        for k in range(nlines * ncols):
            yk, xk = np.unravel_index(k, self.shape)
            self.axes_dims.append((keys[xk], keys[yk]))
            sharey = None
            sharex = None
            if (xk >= 0):
                sharex = self.axes[0, xk]
            if (yk >= 0):
                sharey = self.axes[yk, 0]

            ax = plt.subplot(nlines, ncols, k + 1,
                             sharey=sharey, sharex=sharex)
            ax.xaxis.set_major_locator(MaxNLocator(self.max_n_ticks,
                                                   prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(self.max_n_ticks,
                                                   prune="both"))
            if (xk > 0):
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(self.lbls[yk])
            if (yk < nlines - 1):
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(self.lbls[xk])
            ax.set_visible(False)
            self.axes[yk, xk] = ax

    def _value_from_data(self, key):
        """ Parse a key for existing data in the dataframe. If not found,
        returns the key directly """
        if not isinstance(key, basestring):
            return key
        elif key not in self.data:
            if self.allow_expressions:
                try:
                    return evalexpr(self.data, key)
                except Exception:
                    pass
            return key
        else:
            return self.data[key]

    def evalexpr(self, expr, exprvars=None, dtype=float):
        """ evaluate expression based on the data and external variables
            all np function can be used (log, exp, pi...)

        Parameters
        ----------
        data: dict or dict-like structure
            data frame / dict-like structure containing named columns

        expr: str
            expression to evaluate on the table
            includes mathematical operations and attribute names

        exprvars: dictionary, optional
            A dictionary that replaces the local operands in current frame.

        dtype: dtype definition
            dtype of the output array

        Returns
        -------
        out : np.array
            array of the result
        """
        return evalexpr(self.data, expr, exprvars=exprvars, dtype=dtype)

    def _apply(self, fn, *args, **kwargs):
        """ Apply a function fn to the data

        Parameters
        ----------
        fn: callable or str
            function to apply or use a Plotter function is exists

        Returns
        -------
        r: tuple
            whatever fn returns
        """
        if isinstance(self.data, Group):
            r = []
            for d in self.data:
                if isinstance(fn, basestring):
                    _fn = getattr(d, fn)
                    if _fn is None:
                        raise AttributeError('No function named {0:s}'
                                             .format(fn))
                    r.append(_fn(*args, **kwargs))
                else:
                    r.append(fn(d, *args, **kwargs))
            return r
        else:
            if isinstance(fn, basestring):
                _fn = getattr(self.data, fn)
                if _fn is None:
                    raise AttributeError('No function named {0:s}'.format(fn))
                return _fn(*args, **kwargs)
            else:
                return fn(self.data, *args, **kwargs)

    def map_diag(self, fn, *args, **kwargs):
        """Plot with a univariate function on each diagonal subplot.

        Parameters
        ----------
        func: callable plotting function
            Must take an x array as a positional arguments and draw onto the
            "currently active" matplotlib Axes. There is a special case when
            using a ``hue`` variable and ``plt.hist``; the histogram will be
            plotted with stacked bars.
        only1d: bool
            set to make the function only use the x-axis instead of both.
        """
        n, _ = self.shape
        r = []

        only1d = kwargs.pop('only1d', False)
        nlines, ncols = self.shape

        if only1d:
            for ek, xk in enumerate(self.keys):
                ax = plt.subplot(nlines, ncols, (ncols + 1) * ek + 1,
                                 sharex=self.axes[0][ek])
                self.axes[ek, ek] = ax
                plt.sca(ax)
                ax.set_visible(True)
                ax.set_xlabel(xk)
                r.append(self._apply(fn, xk, *args, **kwargs))
        else:
            for ek, xk in enumerate(self.keys):
                ax = self.axes[ek, ek]
                plt.sca(ax)
                ax.set_visible(True)
                r.append(self._apply(fn, xk, xk, *args, **kwargs))

        self._check_label_visibility()
        return r

    def map_offdiag(self, fn, *args, **kwargs):
        """Plot with a bivariate function on off-diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.
        """
        r = []
        for ax, (xk, yk) in zip(np.ravel(self.axes), self.axes_dims):
            if xk != yk:
                plt.sca(ax)
                ax.set_visible(True)
                r.append(self._apply(fn, xk, yk, *args, **kwargs))
        self._check_label_visibility()
        return r

    def map_lower(self, fn, *args, **kwargs):
        """Plot with a bivariate function on the lower diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.
        """
        lbl_k = -1
        nlines, ncols = self.shape
        r = []
        for yi in range(nlines):
            for xi in range(ncols):
                lbl_k += 1
                if xi < yi:
                    ax = self.axes[yi, xi]
                    plt.sca(ax)
                    ax.set_visible(True)
                    xk, yk = self.axes_dims[lbl_k]
                    r.append(self._apply(fn, xk, yk, *args, **kwargs))
        self._check_label_visibility()
        return r

    def map_upper(self, fn, *args, **kwargs):
        """Plot with a bivariate function on the upper diagonal subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.
        """
        lbl_k = -1
        nlines, ncols = self.shape
        r = []
        for yi in range(nlines):
            for xi in range(ncols):
                lbl_k += 1
                if xi > yi:
                    ax = self.axes[yi, xi]
                    plt.sca(ax)
                    ax.set_visible(True)
                    xk, yk = self.axes_dims[lbl_k]
                    r.append(self._apply(fn, xk, yk, *args, **kwargs))
        self._check_label_visibility()
        return r

    def map(self, fn, *args, **kwargs):
        """Plot with a bivariate function on all subplots.

        Parameters
        ----------
        func : callable plotting function
            Must take x, y arrays as positional arguments and draw onto the
            "currently active" matplotlib Axes.
        """
        r = self.map_offdiag(fn, *args, **kwargs)
        r.extend(self.map_diag(fn, *args, **kwargs))
        return r


class CornerPlot():
    """ Generates a corner plot rapidly

    Attributes
    ----------
    plotter: Plotter instance
        plotter used by the class.

    pp: PairGrid instance
        Pair grid that will be used to make the plots
    """
    def __init__(self, plotter, varnames=None, labels=None,
                 figsize=None, **kwargs):
        """ Constructor

        Parameters
        ----------
        plotter: Plotter instance
            plotter to use. If a dataframe is provided, the default will be to
            use Plotter(plotter).

        varnames: seq(str)
            limit the plot to a subset of variables

        labels: seq(str)
            replace the variable names by provided labels

        figsize: tuple(height, width)
            Size of the figure. Default is a square of length 3 * len(varnames)

        kwargs: dict
            Forwarded to `Plotter.pairplot` method
        """
        if isinstance(plotter, Plotter):
            self.plotter = plotter
        else:
            self.plotter = Plotter(plotter)

        if varnames is None:
            varnames = list(self.plotter.keys)

        if labels is None:
            labels = varnames

        if (figsize is None):
            figsize = (3 * len(varnames), 3 * len(varnames))

        plt.figure(figsize=figsize)
        self.pp = plotter.pairplot(varnames, labels=labels, **kwargs)

    def diag(self, fn=None, **kwargs):
        """ Make the diagonal plot using fn """
        defaults = dict(only1d=True, bins=32, edgecolor='k',
                        facecolor='None', histtype='step')
        if fn is None:
            defaults = dict(only1d=True, bins=32, edgecolor='k',
                            facecolor='None', histtype='step')
            defaults.update(kwargs)
            self.pp.map_diag('hist', **defaults)
        else:
            defaults = dict(only1d=True)
            defaults.update(kwargs)
            self.pp.map_diag(fn, **defaults)
        return self

    @property
    def data(self):
        """ Get the dataframe directly """
        return self.pp.data.data

    def add_quantiles(self, quantiles=[0.16, 0.5, 0.84]):
        """ Adds quantile indications on the diagonal plots """
        for num, (kx, labelx) in enumerate(zip(self.pp.keys, self.pp.lbls)):
            ax = self.pp.axes[num][num]
            q_16, q_50, q_84 = np.quantile(self.data[kx], quantiles)
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
        return self

    def lower(self, fn='plot', **kwargs):
        """ Makes the lower diagonal plots """
        self.pp.map_lower('plot', **kwargs)
        return self

    def upper(self, fn='plot', **kwargs):
        """ Makes the upper diagonal plots """
        self.pp.map_upper('plot', **kwargs)
        return self


def _intercept_empty_plot(*args, **kwargs):
    """ fall back to empty plot when data is empty

    Mostly designed to produce plots when forced group selections are made
    """
    ax = kwargs.pop('ax', None)
    if ax is None:
        ax = plt.gca()
    # ax.cla()
    ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
            transform=ax.transAxes)
    # Adjust x, y ticks spacing.
    # plt.setp(ax.get_xticklabels() + ax.get_yticklabels(), visible=False)
    # plt.setp(ax.get_xticklines() + ax.get_yticklines(), visible=False)


def evalexpr(data, expr, exprvars=None, dtype=float):
    """ evaluate expression based on the data and external variables
        all np function can be used (log, exp, pi...)

    Parameters
    ----------
    data: dict or dict-like structure
        data frame / dict-like structure containing named columns

    expr: str
        expression to evaluate on the table
        includes mathematical operations and attribute names

    exprvars: dictionary, optional
        A dictionary that replaces the local operands in current frame.

    dtype: dtype definition
        dtype of the output array

    Returns
    -------
    out : np.array
        array of the result
    """
    _globals = {}
    keys = []
    if hasattr(data, 'keys'):
        keys += list(data.keys())
    if hasattr(getattr(data, 'dtype', None), 'names'):
        keys += list(data.dtype.names)
    if hasattr(data, '_aliases'):
        # SimpleTable specials
        keys += list(data._aliases.keys())
    keys = set(keys)
    for k in keys:
        if k in expr:
            _globals[k] = data[k]

    if exprvars is not None:
        if (not (hasattr(exprvars, 'items'))):
            msg = "Expecting a dict-like as condvars with an `items` method"
            raise AttributeError(msg)
        for k, v in (exprvars.items()):
            _globals[k] = v

    # evaluate expression, to obtain the final filter
    # r = np.empty( self.nrows, dtype=dtype)
    r = eval(expr, _globals, np.__dict__)

    return np.array(r, dtype=dtype)


def create_common_cbar(vmin=0, vmax=1, box=None, **kwargs):
    """ Create a common colorbar to a complex figure

    Parameters
    ----------
    vmin: float
        minimum value on the colorscale
    vmax: float
        maximum value on the colorscale
    box: tuple
        axis definition box

    Returns
    -------
    cb: ColorBar instance
        the colorbar object
    """
    if box is None:
        box = [0.3, 0.1, 0.6, 0.02]

    kw = dict(spacing='proportional', orientation='horizontal',
              cmap=plt.cm.jet)

    kw.update(**kwargs)
    norm = kw.pop('norm', None)
    if norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    ax = plt.gcf().add_axes(box)
    cb = mpl.colorbar.ColorbarBase(ax, norm=norm, **kw)
    return cb


def create_common_legend(labels, colors, markers='s', mec=None,
                         linestyles='None', linewidths=None, fig=None,
                         **kwargs):
    """ Create a legend from the symbols without the actual plot

    Parameters
    ----------
    labels: seq
        sequence of label strings
    colors: seq or Colormap
        sequence of colors or Colormap instance from which deriving a
        sequence of colors to encode each group
        if Colormap instance, a cmap attribute will be generated after a
        plot and will refer to the updated instance
    markers: seq
        sequence of markers (will cycle through)
        default is `s`, i.e., a square
    mec: seq
        marker edge colors
    linestyles: seq
        sequence of linestyles (will cycle through)
    linewidths: seq
        sequence of linewidths (will cycle through)
    fig: plt.Figure
        figure to add a legend (default: `plt.gcf()`)
    kwargs: dict
        any other keyword will go to :func:`plt.legend`

    Returns
    -------
    lgd: plt.Legend instance
        the newly created legend
    """
    from matplotlib.lines import Line2D
    from itertools import cycle

    if fig is None:
        fig = plt.gcf()

    defaults = dict(numpoints=1, frameon=False)
    defaults.update(kwargs)

    if not hasattr(mec, '__iter__'):
        mec = [mec]
    if not hasattr(linewidths, '__iter__'):
        linewidths = [linewidths]

    lines = []
    for lbl, color, m, ls, me, lw in zip(labels, colors, cycle(markers),
                                         cycle(linestyles), cycle(mec),
                                         cycle(linewidths)):
        line_ = Line2D(range(2), range(2), marker=m, mec=me, linestyle=ls,
                       color=color, lw=lw)
        lines.append(line_)

    lgd = fig.legend(lines, labels, **defaults)
    plt.draw_if_interactive()
    return lgd


def colorify(data, vmin=None, vmax=None, cmap=plt.cm.Spectral):
    """ Associate a color map to a quantity vector

    Parameters
    ----------
    data: sequence
        values to encode
    vmin: float
        minimum value
    vmax: float
        maximum value
    cmap: Colormap instance
        colormap to use

    Returns
    -------
    colors: sequence or array
        one color per input data
    cmap: Colormap
        data normalized colormap instance
    """
    try:
        from matplotlib.colors import Normalize
    except ImportError:
        # old mpl
        from matplotlib.colors import normalize as Normalize

    _vmin = vmin or min(data)
    _vmax = vmax or max(data)
    cNorm = Normalize(vmin=_vmin, vmax=_vmax)

    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=cmap)
    try:
        colors = scalarMap.to_rgba(data)
    except Exception:
        colors = list(map(scalarMap.to_rgba, data))
    scalarMap.set_array(data)
    return colors, scalarMap
