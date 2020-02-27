"""
Uses Datashader to plot things
"""
from __future__ import absolute_import
import datashader as ds
import numpy as np
import pylab as plt

import matplotlib.image as mimage
from matplotlib.transforms import (Bbox, TransformedBbox, BboxTransform)
from matplotlib import colors
from .matplotlib import norm as eznorm

from .plotter import Plotter
from .dask import dummy

_ACCEPTED_DF_TYPE = []
try:
    import pandas as pd_
    _ACCEPTED_DF_TYPE.append(pd_.DataFrame)
except ImportError:
    pd_ = None

try:
    import dask.dataframe as dd_
    _ACCEPTED_DF_TYPE.append(dd_.DataFrame)
except ImportError:
    pass


class DSArtist(mimage._ImageBase):
    """ Artist view to interface datashader canvas with Matplotlib """

    def __init__(self, data, xname, yname,
                 agg=None, kind='points', ax=None, spread=1,
                 **kwargs):
        """ Constructor """
        if ax is None:
            ax = plt.gca()

        self.vmin = kwargs.pop('vmin', None)
        self.vmax = kwargs.pop('vmax', None)
        kwargs['norm'] =  self.parse_norm(norm=kwargs.pop('norm', None))
        self.alpha_below = kwargs.pop('alpha_below', None)
        super().__init__(ax, **kwargs)

        self.kind = kind
        self.data = data
        self.xname = xname
        self.yname = yname
        self.agg = self.parse_agg(agg=agg)
        self.axes = ax
        self.spread = spread
        ax.set_ylim((np.nanmin(data[yname]), np.nanmax(data[yname])))
        ax.set_xlim((np.nanmin(data[xname]), np.nanmax(data[xname])))
        self.set_array([[1, 1], [1, 1]])
        
    def parse_agg(self, **kwargs):
        """ Allows one to use a string shortcut to defined the agg keyword
        
        e.g.: parse_agg('mean(z)'), parse_agg('var(x + y)')
        
        Parameters
        ----------
        kwargs: dict
            keywords given to the artist
         
        Returns
        -------
        agg: datashader.reduction function
        """
        if agg is not None:
            try:
                name = ''
                for k in agg:
                    if k != '(':
                        name += k
                    else:
                        break
                rest_ = agg.replace(name + '(', '').replace(')', '')
                fn_ = ['any', 'count', 'sum', 'min', 'max', 'count_cat', 
                       'mean', 'var', 'std', 'first', 'last', 'mode']
                mapped = {k: getattr(datashader.reductions, k) for k in fn_}
                agg_ = mapped.get(name, None)
                return agg_(rest_)
            except Exception as e:
                return agg
    
    def parse_norm(self, **kwargs):
        """ Allows one to use a string shortcut to defined the norm keyword
        
        e.g.: parse_norm('log10')
        
        Parameters
        ----------
        kwargs: dict
            keywords given to the artist
         
        Returns
        -------
        norm: colors.Normalize object
        """
        norm = kwargs.pop('norm', None)
        if norm is not None:
            mapped = {'arcsinh': eznorm.Arcsinh,
                      'log10': colors.LogNorm,
                      'sqrt': eznorm.Sqrt,
                      'pow': eznorm.Power,
                      'histeq': eznorm.HistEq,
                      'midpoint': eznorm.MidpointNormalize
                      }
            try:
                norm_ = eval(norm, mapped, colors.__dict__)
                if isinstance(norm_, type):
                    if norm_ == eznorm.HistEq:
                        return norm_(self)
                    return norm_()
                else:
                    return norm_
            except Exception as e:
                print(e)
                return norm
            
    def set_norm(self, norm):
        """ update norm """
        self.update({'norm': self.parse_norm(norm='histeq')})
        self.changed()

    def make_image(self, renderer, magnification=1.0,
                   unsampled=False):
        """ Generate the image content """
        trans = self.get_transform()
        # (x_1, x_2), (y_1, y_2) = self.axes.get_xlim(), self.axes.get_ylim()
        x_1, x_2, y_1, y_2 = self.get_extent()
        bbox = Bbox(np.array([[x_1, y_1], [x_2, y_2]]))
        transformed_bbox = TransformedBbox(bbox, trans)

        dims = self.axes.patch.get_window_extent().bounds
        plot_width = int(dims[3] + 0.5) // self.spread
        plot_height = int(dims[2] + 0.5) // self.spread

        x_range = min(x_1, x_2), max(x_1, x_2)
        y_range = min(y_1, y_2), max(y_1, y_2)

        cvs = ds.Canvas(plot_width=plot_width,
                        plot_height=plot_height,
                        x_range=x_range, y_range=y_range)
        ds_func = getattr(cvs, self.kind, self.kind)
        img = ds_func(self.data, self.xname, self.yname, self.agg)
        if x_1 > x_2:
            img = np.fliplr(img)
        if y_1 < y_2:
            img = np.flipud(img)
        img = np.ma.masked_invalid(img)
        vmin = self.vmin or np.nanmin(img)
        vmax = self.vmax or np.nanmax(img)
        img.clip(vmin, vmax, out=img)
        self.set_clim(np.nanmin(img) + 1 * (self.vmin is None), np.nanmax(img))
        if self.alpha_below is not None:
            img = np.ma.masked_less_equal(img, self.alpha_below, copy=False)
        self.set_data(img)
        return self._make_image(img, bbox, transformed_bbox, self.axes.bbox,
                                magnification, unsampled=unsampled)

    @property
    def _extent(self):
        return self.get_extent()

    def get_extent(self):
        """ returns the image extension """
        x_1, x_2 = self.axes.get_xlim()
        y_1, y_2 = self.axes.get_ylim()
        return (x_1, x_2, y_1, y_2)

    def get_cursor_data(self, event):
        """Get the cursor data for a given event"""
        xmin, xmax, ymin, ymax = self.get_extent()
        if self.origin == 'upper':
            ymin, ymax = ymax, ymin
        arr = self.get_array()
        data_extent = Bbox([[ymin, xmin], [ymax, xmax]])
        array_extent = Bbox([[0, 0], arr.shape[:2]])
        trans = BboxTransform(boxin=data_extent,
                              boxout=array_extent)
        yval, xval = event.ydata, event.xdata
        i, j = trans.transform_point([yval, xval]).astype(int)
        # Clip the coordinates at array bounds
        if (not 0 <= i < arr.shape[0]) or (not 0 <= j < arr.shape[1]):
            return None

        return arr[i, j]


class DSPlotter(Plotter):
    """ Extended Plotter to exploit Datashader """

    def get_dataframe(self, *args):
        """ Get the data in a format accepted by datashader

        args: seq(str)
            names of the relevant variables
        """
        if pd_ is None:
            raise ImportError("You need at least pandas to use this class")
        if isinstance(self.data, tuple(_ACCEPTED_DF_TYPE)):
            return dummy(self.data)
        try:
            # one of SimpleTable or DictDataFrame maybe?
            df_ = self.data.to_pandas(keys=args)
        except AttributeError:
            df_ = pd_.DataFrame.from_dict(
                {name: self._value_from_data(name) for name in args}
                )
        return df_

    def plot(self, xname, yname, agg=None, **kwargs):
        """ Plotting standard call

        Parameters
        ----------
        xname: str
            xname or variable expression

        yname: str
            yname or variable expression

        agg: ds.aggregator instance
            how to aggregate the data (default: count())

        kwargs: dict
            passed to the various actors

        returns
        -------
        da: DSArtist instance
            matplotlib artist image used to plot the data
        """
        kwargs.setdefault('kind', 'points')
        if agg is None:
            agg = ds.count()

        other_names = []
        for input_k in agg.inputs:
            other_names.extend(input_k.inputs)

        artist = DSArtist(self.get_dataframe(xname, yname, *other_names),
                          xname, yname, agg=agg, **kwargs)
        artist.axes.add_artist(artist)
        plt.sca(artist.axes)
        self._set_auto_axis_labels(xname, yname, ax=artist.axes)
        return artist

    def line(self, xname, yname, agg=None, **kwargs):
        """ Plot a line """
        kwargs['kind'] = 'line'
        return self.plot(xname, yname, agg=agg, **kwargs)

    def points(self, xname, yname, agg=None, **kwargs):
        """ Plot a line """
        kwargs['kind'] = 'points'
        return self.plot(xname, yname, agg=agg, **kwargs)
