""" Wrapper of h5py to dask.dataframe

This package allows one to load HDF5 table into a dask.dataframe regardless of
the structure of it.
"""
from __future__ import absolute_import, division, print_function
from math import ceil
from glob import glob
import os

import numpy as np
import pandas as pd
import h5py

from dask.base import tokenize
import dask.dataframe as dd


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


def read_table(filepath, grouppath='/', keys=None, chunksize=int(10e6),
               index=None, lock=None):
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
    print(glob_)
    
    if len(glob_) > 1:
        dfs = [read_table(name_k, grouppath=grouppath, keys=keys, 
                          chunksize=chunksize, index=index, lock=lock) 
               for name_k in glob_]
        return dd.concat(dfs, interleave_partitions=True)
    else:
        filepath = glob_[0]
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath + ' does not seem to exist.')
    
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
                    index=None, lock=None):
    """
    Shortcut to :py:func:`read_table` 
    where the default grouppath is set to Vaex format.

    Returns
    -------
    :py:class:`dask.dataframe.DataFrame`
    """
    return read_table(filepath, grouppath=grouppath, keys=keys, 
                      chunksize=chunksize, index=index, lock=lock)
