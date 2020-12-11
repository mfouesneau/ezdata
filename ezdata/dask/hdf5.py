""" Wrapper of h5py to dask.dataframe

This package allows one to load HDF5 table into a dask.dataframe regardless of
the structure of it.
"""
from __future__ import absolute_import, division, print_function
from math import ceil
from glob import glob
from collections import OrderedDict
import os

import numpy as np
import pandas as pd
import h5py

from dask.base import tokenize
import dask.dataframe as dd
try:
    from tqdm import tqdm
except:
    # if not present still works
    tqdm = lambda x: x


def _get_columns(grp, key):
    """ Get data columns regardless of wether grp or grp/data is the array"""
    if isinstance(grp[key], h5py.Group):
        return grp[key + '/data']
    return grp[key]


def _get_group_info(path, grouppath, keys):
    """ Get metadata about a group in the given file

    Parameters
    ----------
    path: str
        path to hdf5 file
    grouppath: str
        which group
    keys: seq(str)
        which columns to read

    Returns
    -------
    nrows: int
        number of data entries
    keys:
    meta:
    categoricals:
    """
    with h5py.File(path, "r") as input_file:
        grp = input_file[grouppath]

        if keys is None:
            keys = list(grp.keys())

        categoricals = {}
        for key in keys:
            dtype_ = h5py.check_dtype(enum=_get_columns(grp, key).dtype)
            if dtype_ is not None:
                categoricals[key] = sorted(dtype_, key=dtype_.__getitem__)

        # Meta is an empty dataframe that serves as a compound "dtype"
        meta = pd.DataFrame(
            {key: np.array([], dtype=_get_columns(grp, key).dtype)
             for key in keys},
            columns=keys)

        for key in categoricals:
            meta[key] = pd.Categorical([], categories=categoricals[key],
                                       ordered=True)
        nrows = len(_get_columns(grp, keys[0]))
    return nrows, keys, meta, categoricals


def _slice_dataset(filepath, grouppath, key, slc, lock=None):
    """ Get a slice of the dataset """
    try:
        if lock is not None:
            lock.acquire()
        with h5py.File(filepath, "r") as input_file:
            return _get_columns(input_file[grouppath], key)[slc]
    finally:
        if lock is not None:
            lock.release()


def _slice_group(filepath, grouppath, keys, slc, lock=None):
    """ Get a slice of a given group """
    try:
        if lock is not None:
            lock.acquire()
        with h5py.File(filepath, "r") as input_file:
            return {key: _get_columns(input_file[grouppath], key)[slc]
                    for key in keys}
    finally:
        if lock is not None:
            lock.release()


def _restore_categories(data, categorical_columns):
    """ Restore categories the data """
    for key, category_dict in categorical_columns.items():
        data[key] = pd.Categorical.from_codes(data[key], category_dict,
                                              ordered=True)
    return data


class _H5Collector:
    """
    Extract shapes and dtypes of all array objects in a give hdf5 file
    It does so recursively and only reports array objects

    Allows one to make statistics and checks, which are necessary to
    concatenate datasets.

    Properties
    ----------
    names: dict
        field names and shapes

    dtypes: dict
        contains the dtype of the registered names
    """
    def __init__(self):
        """ Constructor """
        # Store the columns and shapes in order
        self.names = OrderedDict()
        self.dtypes = {}

    def __repr__(self):
        """ Representation """
        max_key_length = max([len(k) for k in self.names.keys()])
        fmt = ('{key:>' + str(max_key_length) + 's}: {dtype:10s} {shape}')
        text = [fmt.format(key=key, shape=shape, dtype=str(self.dtypes[key]))
                for key, shape in self.names.items()]
        return '\n'.join(text)

    def __call__(self, name, h5obj):
        """ apply the collector to a new object.
        This method is called by `h5py.File.visititems`
        within `_H5Collector.add`
        """
        # only h5py datasets have dtype attribute, so we can search on this
        if hasattr(h5obj, 'dtype') and hasattr(h5obj, 'shape'):

            if name not in self.dtypes:
                self.dtypes[name] = h5obj.dtype
            elif (self.dtypes[name].char == 'S') & (h5obj.dtype.char == 'S'):
                # String length updates
                dt_size = max(1, max(self.dtypes[name].itemsize, h5obj.dtype.itemsize))
                self.dtypes[name] = np.dtype('S' + str(dt_size))
            elif self.dtypes[name] != h5obj.dtype:
                raise RuntimeError('Type mismatch in {0:s}'.format(name))
            try:
                shape_x, shape_y = h5obj.shape
                shape = self.names.get(name, (0, shape_y))
                if shape_y != shape[1]:
                    raise RuntimeError('Shape mismatch in {0:s}'.format(name))
                self.names[name] = shape[0] + shape_x, shape_y
            except ValueError:
                shape_x, = h5obj.shape
                shape, = self.names.get(name, (0,))
                self.names[name] = (shape + shape_x, )

    def add(self, filename):
        """ Add filename to the collection

        Parameters
        ----------
        filename : str
            file to add to the collection

        Returns
        -------
        self: _H5Collector
            itself
        """
        with h5py.File(filename, 'r') as datafile:
            datafile.visititems(self)
        return self


def _ignore_multidimensional_keys(filename, grouppath=None):
    """ Check keys to make sure not multi-dimensional arrays are provided """
    hls = _H5Collector()
    if grouppath is not None:
        with h5py.File(filename, 'r') as datafile:
            datafile[grouppath].visititems(hls)
    else:
        hls.add(filename)

    keys = [name.replace('/data', '')
            for (name, shape) in hls.names.items() if len(shape) < 2]
    return keys


def read_table(filepath, grouppath='/', keys=None, chunksize=int(10e6),
               index=None, lock=None, ignore_nd_data=True):
    """
    Create a dask dataframe around a column-oriented table in HDF5.

    A table is a group containing equal-length 1D datasets.

    Parameters
    ----------
    filepath: str, seq(str)
        path to the filename or pattern to the tables to open at once.
        This may be also a sequence of files that will be concatenated.
    grouppath : str
        tree path to the HDF5 group storing the table.
    keys : list, optional
        list of HDF5 Dataset keys, default is to use all keys in the group
    chunksize : int, optional
        Chunk size
    index : str, optional
        Sorted column to use as index
    lock : multiprocessing.Lock, optional
        Lock to serialize HDF5 read/write access. Default is no lock.
    ignore_nd_data: bool, optional
        Set to safely ignore keys of multidimensional data arrays
        Note that dask/pandas DataFrame do not support multidimensional data

    Returns
    -------
    :py:class:`dask.dataframe.DataFrame`

    Notes
    -----
    Learn more about the `dask <https://docs.dask.org/en/latest/>`_ project.

    """
    # handle pattern input
    try:
        glob_ = glob(filepath)
    except TypeError:
        glob_ = filepath

    if len(glob_) > 1:
        dfs = [read_table(name_k, grouppath=grouppath, keys=keys,
                          chunksize=chunksize, index=index, lock=lock)
               for name_k in glob_]
        return dd.concat(dfs, interleave_partitions=True)
    else:
        filepath = glob_[0]

    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath + ' does not seem to exist.')

    if ignore_nd_data:
        keys_1d = _ignore_multidimensional_keys(filepath, grouppath)
    if keys is None:
        keys = keys_1d
    else:
        keys = [key for key in keys if key in keys_1d]

    nrows, keys, meta, categoricals = _get_group_info(filepath,
                                                      grouppath,
                                                      keys)
    # Make a unique task name
    token = tokenize(filepath, grouppath, chunksize, keys)
    task_name = "daskify-h5py-table-" + token

    # Partition the table
    divisions = (0,) + tuple(range(-1, nrows, chunksize))[1:]
    if divisions[-1] != nrows - 1:
        divisions = divisions + (nrows - 1,)

    # Build the task graph
    dsk = {}
    for i in range(0, int(ceil(nrows / chunksize))):
        slc = slice(i * chunksize, (i + 1) * chunksize)
        data_dict = (_slice_group, filepath, grouppath, keys, slc, lock)
        if categoricals:
            data_dict = (_restore_categories, data_dict, categoricals)
        dsk[task_name, i] = (pd.DataFrame, data_dict, None, meta.columns)

    # Generate ddf from dask graph
    _df = dd.DataFrame(dsk, task_name, meta, divisions)
    if index is not None:
        _df = _df.set_index(index, sorted=True, drop=False)
    return _df


def read_vaex_table(filepath, grouppath='/table/columns',
                    keys=None, chunksize=int(10e6),
                    index=None, lock=None, ignore_nd_data=True):
    """
    Shortcut to :py:func:`read_table`
    where the default grouppath is set to Vaex format.

    Returns
    -------
    :py:class:`dask.dataframe.DataFrame`
    """
    return read_table(filepath, grouppath=grouppath, keys=keys,
                      chunksize=chunksize, index=index, lock=lock)


def concatenate(*args, **kwargs):
    """ Concatenate multiple HDF5 files with the same structure

    This routine is the most flexible I could make. It takes any datashapes
    (contrary to vaex, pandas, dask etc) and copies the data into to final
    output file.

    Parameters
    ----------
    args: seq(str)
        filenames to concatenate

    pattern: str, optional
        pattern of files to concatenate

    outputfile: str, optional
        filename of the output file containing the data

    verbose: bool, optional
        set to display information

    returns
    -------
    outputfile: str
        the filename of the result
    """
    pattern = kwargs.get('pattern', None)
    if (pattern is None) and (not args):
        raise RuntimeError('Must provide either a pattern or a list of files')
    if not args:
        args = glob(pattern)

    output = kwargs.get('outputfile', None)
    if output is None:
        output = '.'.join(args[0].split('.')[:-1]) + '_concat.hdf5'

    verbose = kwargs.get('verbose', False)

    def info(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    info('Collecting information from {0:d} files'.format(len(args)))
    hls = _H5Collector()
    for fname in tqdm(args):
        hls.add(fname)

    with h5py.File(output, 'w') as outputfile:

        # creating the final file with all empty structure
        info('Creating {0:s} with empty structure'.format(output))
        for name, shape in hls.names.items():
            group_name = name.split('/')
            group_ = '/'.join(group_name[:-1])
            name_ = group_name[-1]
            dtype = hls.dtypes[name]
            outputfile.create_group(group_)\
                      .create_dataset(name_, shape=shape, dtype=dtype)

        # copy the data over
        index = 0
        info('Copying data')
        for iternum, fname in enumerate(tqdm(args), 1):
            with h5py.File(fname, 'r') as fin:
                keys = list(hls.names.keys())
                length = len(fin[keys[0]])
                for name in hls.names.keys():
                    data = fin[name]
                    length = len(data)
                    outputfile[name][index: length + index] = data[:]
            index += length
            info('... [{0:d} / {1:d}] - done with {2:s}'.format(iternum, len(args), fname))

    return output


def to_vaex_file(ds, output, grouppath='/table/columns',
                 keys=None, **kwargs):
    """
    export a dask dataframe into a vaex formatted hdf5 file.

    Parameters
    ----------
    ds: dask.DataFrame
        data to export

    output: str
        filename of the exported table

    grouppath: str
        vaex default path to the dataset

    keys: sequence(str)
        subset of columns to export (default all)

    verbose: bool
        set to have information messages
    """
    verbose = kwargs.get('verbose', False)

    def info(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    dtypes = ds.dtypes.to_dict()
    if keys is not None:
        dtypes = dict(((name, dtypes[name]) for name in keys))

    for name, dtype in dtypes.items():
        if 'numpy.object_' in str(dtype.type):
            # we have a pandas string that does not work well with h5py
            maxlen = ds[name].dropna().str.len().max().compute().astype(int)
            col_type = np.dtype('{0:s}{1:d}'.format('S', maxlen))
            dtypes[name] = col_type
            info('Object type conversion: "{0:s}" as "{1:s}"'.format(
                name, str(col_type)))

    length = ds.shape[0].compute()

    def construct_vaex_path(name):
        path = '{grouppath:s}/{name:s}/data'
        return path.format(name=name, grouppath=grouppath)

    with h5py.File(output, 'w') as outputfile:

        # creating the final file with all empty structure
        info('Creating {0:s} with empty structure'.format(output))
        for name, dtype in dtypes.items():
            group_name = construct_vaex_path(name).split('/')
            group_ = '/'.join(group_name[:-1])
            name_ = group_name[-1]

            outputfile.create_group(group_)\
                      .create_dataset(name_, shape=(length,), dtype=dtype)

        # copy the data over
        index = 0
        info('Copying data')

        names = dtypes.keys()

        for part_i in range(ds.npartitions):
            df = ds.get_partition(part_i).compute()
            df_size = df.shape[0]
            for name in names:
                vaex_name = construct_vaex_path(name)
                data = df[name].values.astype(dtypes[name])
                outputfile[vaex_name][index: df_size + index] = data[:]
            index += df_size
            info('... [{0:d} / {1:d}] partition done'.format(part_i + 1,
                                                             ds.npartitions))
        return output
