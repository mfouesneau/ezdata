# ezData - A simplistic column based data framework. 

> tested with python 2.7, & 3.4, 3.5, 3.6, 3.7
>
> compatible with many existing dataframes: e.g. pandas 
>
> requirements: numpy, matplotlib (for plotting only)
>                conversion to other formats require the appropriate library.

.. notes::

    * requirements: numpy, matplotlib
    * conversion to other formats require the appropriate library.
    * some additional wrappers around bokeh, plotly and holoviews are in dev.

:author: Morgan Fouesneau

Documentation and API: [link](http://mfouesneau.github.io/docs/ezdata/)

Example notebook in the `examples` directory

[![nbviewer](https://img.shields.io/badge/render%20on-nbviewer-orange.svg](https://nbviewer.jupyter.org/github/mfouesneau/ezdata/blob/master/examples/examples.ipynb)

## Why?

I always found myself writing snippets around `numpy`, `matplotlib`, pandas and
other file readers. These are often the same things: read file `foo` and plot
`a` against `b` where `something is takes some values`. 
It gets always very complex when you want to make something non-standard, for
instance, _for each of the 10 classes given according to this selection, make a
scatter plot with these specific markers and color coded by another column_.

_I was basically tired of all the packages doing fancy things and not allowing
basics or requiring a lot of dependencies._

This package initially focused on easily manipulating column oriented data.
In particular this package allows easy conversions to many common dataframe
containers: `dict`, `numpy.recarray`, `pandas.DataFrame`, `dask.DataFrame`,
`astropy.Table`, `xarray.Dataset`, `vaex.DataSetArrays`.

I extended this package to allow myself to plot these data in the a very simple 
manner. Of course this was not covering all needs and thus I added interfaced to
`holoviews/datashader`.

## What is this package?

Based on the most basic functions and in particular methods of `dict`, I wrote
this package. This basically builds advance-ish access to column oriented data
through 4 main classes, 2 of which handle data and the others plotting shortcuts. 
**This may not fit all needs, nor large data access**.

* `dictdataframe`: an advanced dictionary object.
	A simple-ish dictionary like structure allowing usage as array on non
	constant multi-dimensional column data.  The :class:`DataFrame`
	container allows easier manipulations of the data but is basically a
	wrapper of many existing function around a `dictionary` object.

* `simpletable`: a simplified version of [ezTables](https://github.com/mfouesneau/eztables)
	The :class:`SimpleTable` allows easier manipulations of the data
	but is basically a wrapper of many existing function around a `numpy.recarray` object.
	It implements reading and writing ascii, FITS and HDF5 files.
	The :class:`AstroTable` built on top of the latter class, adds-on
	astronomy related functions, such as `conesearch`

* `plotter`: this package implements :class:`Plotter`, which is a simple
  container to dictionary like structure (e.g. :class:`dict`,
  :class:`np.recarray`, :class:`pandas.DataFrame`, :class:`SimpleTable`). 
  It allows the user to plot directly using keys of the data and also allows
  rapid group plotting routines (`groupy` and `facets`). Note that is also
  allows expressions instead of keys.  **This interface should basically work on
  any dictionary like structure**
  
* `DSPlotter`:  extends :class:`Plotter`, to use `datashader` for some plots
  allows expressions instead of keys.  **This interface requires `holoview` and `datashader`**

Both data structures implements common ground base to line and column access in
the same transparent manner.  These objects implement for instance array
slicing, shape, dtypes on top of which they implement functions such as:
`sortby`, `groupby`, `where`, `join` and evaluation of expressions as keys. (see
examples below). Both also have a direct access to a `Plotter` attribute. 
`DSPlotter` is experimental and requires more than basic libraries so that it 
needs to be called on the side.

The data classes allows easy conversions to many common dataframe
containers: `numpy.recarray`, `pandas.DataFrame`, `dask.DataFrame`,
`astropy.Table`, `xarray.Dataset`, `vaex.DataSetArrays`.

## Examples

* Some data manipulation basics

```python
    >>> t = SimpleTable('path/mytable.csv')
    # get a subset of columns only
    >>> s = t.get('M_* logTe logLo U B V I J K')
    # set some aliases
    >>> t.set_alias('logT', 'logTe')
    >>> t.set_alias('logL', 'logLLo')
    # make a query on one or multiple column
    >>> q = s.selectWhere('logT logL', '(J > 2) & (10 ** logT > 5000)')
    # note that `q` is also a table object
    # makes a simple plot (see :module:`plotter`)
    >>> q.Plotter.plot('logT', 'logL', ',')
    # export the initial subtable to a new file
    >>> s.write('newtable.fits')
    # or 
    >>> s.write('newtable.hd5')
```

* Convert to other dataframe structures

```python
    >>> t = SimpleTable('path/mytable.csv')
    >>> t.to_pandas()
    >>> t.to_dask(npartitions=5)
    >>> d = DictDataFrame(t)
```

* Make a single plot of 'RA', 'DEC' on which each region 'BRK' is represented by
  a different color (colormap or other) and different marker.

<img src="http://mfouesneau.github.io/docs/ezdata/ex1.png" width="50%">

```python

    >>> p = t.Plotter.groupby('BRK', markers='<^>v.oxs', colors='parula')
    >>> p.plot('CRA', 'CDEC', 'o')
    >>> import pylab as plt
    >>> plt.legend(loc='best', numpoints=1)
    >>> plt.xlim(plt.xlim()[::-1])
    >>> plt.xlabel('RA')
    >>> plt.ylabel('DEC')
```

* make a more complex plot: plot the histogram distribution of 'AV' per region
  given by 'BRK', with given color scheme per region value and individual plots
  with shared axis

<img src="http://mfouesneau.github.io/docs/ezdata/ex2.png" width="50%">

```python

    >>> t.Plotter.groupby('BRK', facet=True, \
            colors=plt.cm.parula, sharex=True, \
	    sharey=True).hist('AV', 
	    bins=np.linspace(t.AV.min(), 
	    t.AV.max(), 20), normed=True)
    >>> for ax in plt.gcf().axes[-3:]: ax.set_xlabel('AV')
```
