'''
DictDataFrame, a simplistic column based dataframe

The :class:`DataFrame` container allows easier manipulations of the data but is
basically deriving dictionary objects. In particular it allows easy conversions
to many common dataframe containers: `numpy.recarray`, `pandas.DataFrame`,
`dask.DataFrame`, `astropy.Table`, `xarray.Dataset`, `vaex.DataSetArrays`.

.. notes::

    * tested with python 2.7, & 3.4
    * requirements: numpy
    * conversion to other formats require the appropriate library.

:author: Morgan Fouesneau
'''
from __future__ import (absolute_import, division, print_function)
import numpy as np
import sys
import itertools
import operator

PY3 = sys.version_info[0] > 2

if PY3:
    iteritems = operator.methodcaller('items')
    itervalues = operator.methodcaller('values')
    basestring = (str, bytes)
else:
    range = xrange
    from itertools import izip as zip
    iteritems = operator.methodcaller('iteritems')
    itervalues = operator.methodcaller('itervalues')
    basestring = (str, unicode)

try:
    from .plotter import Plotter
except ImportError:
    Plotter = None


__all__ = ['DictDataFrame']


def pretty_size_print(num_bytes):
    """
    Output number of bytes in a human readable format

    Parameters
    ----------
    num_bytes: int
        number of bytes to convert

    returns
    -------
    output: str
        string representation of the size with appropriate unit scale
    """
    if num_bytes is None:
        return

    KiB = 1024
    MiB = KiB * KiB
    GiB = KiB * MiB
    TiB = KiB * GiB
    PiB = KiB * TiB
    EiB = KiB * PiB
    ZiB = KiB * EiB
    YiB = KiB * ZiB

    if num_bytes > YiB:
        output = '%.3g YB' % (num_bytes / YiB)
    elif num_bytes > ZiB:
        output = '%.3g ZB' % (num_bytes / ZiB)
    elif num_bytes > EiB:
        output = '%.3g EB' % (num_bytes / EiB)
    elif num_bytes > PiB:
        output = '%.3g PB' % (num_bytes / PiB)
    elif num_bytes > TiB:
        output = '%.3g TB' % (num_bytes / TiB)
    elif num_bytes > GiB:
        output = '%.3g GB' % (num_bytes / GiB)
    elif num_bytes > MiB:
        output = '%.3g MB' % (num_bytes / MiB)
    elif num_bytes > KiB:
        output = '%.3g KB' % (num_bytes / KiB)
    else:
        output = '%.3g Bytes' % (num_bytes)

    return output


class DictDataFrame(dict):
    """
    A simple-ish dictionary like structure allowing usage as array on non
    constant multi-dimensional column data.

    It initializes like a normal dictionary and can be used as such.
    A few divergence points though: some default methods such as :func:`len`
    may refer to the lines and not the columns as a normal dictionary would.

    This data object implements also array slicing, shape, dtypes and some data
    functions (sortby, groupby, where, select, etc)
    """
    def __init__(self, *args, **kwargs):
        """ A dictionary constructor and attributes declaration """
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self   # give access to everything directly

    def __len__(self):
        """ Returns the number of rows """
        return len(self[list(self.keys())[0]])

    def to_dict(self, keys=None, contiguous=False):
        """ Construct a dictionary from this dataframe with contiguous arrays

        Parameters
        ----------
        keys: sequence, optional
            ordered subset of columns to export

        contiguous: boolean
            make sure each value is a contiguous numpy array object
            (C-aligned)

        Returns
        -------
        data: dict
            converted data
        """
        if keys is None:
            keys = self.keys()
        if contiguous:
            return {k: np.ascontiguousarray(self[k]) for k in keys}
        return {k: self[k] for k in keys}

    def to_records(self, keys=None):
        """ Construct a numpy record array from this dataframe

        Parameters
        ----------
        keys: sequence, optional
            ordered subset of columns to export

        Returns
        -------
        arr: recarray
            converted data
        """
        return _convert_dict_to_structured_ndarray(self, keys=keys)

    def to_pandas(self, **kwargs):
        """ Construct a pandas dataframe

        Parameters
        ----------
        data : ndarray
            (structured dtype), list of tuples, dict, or DataFrame
        keys: sequence, optional
            ordered subset of columns to export
        index : string, list of fields, array-like
            Field of array to use as the index, alternately a specific set of
            input labels to use
        exclude : sequence, default None
            Columns or fields to exclude
        columns : sequence, default None
            Column names to use. If the passed data do not have names
            associated with them, this argument provides names for the
            columns. Otherwise this argument indicates the order of the columns
            in the result (any names not found in the data will become all-NA
            columns)
        coerce_float : boolean, default False
            Attempt to convert values to non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets

        Returns
        -------
        df : DataFrame
        """
        try:
            from pandas import DataFrame
            keys = kwargs.pop('keys', None)
            return DataFrame.from_dict(self.to_dict(keys=keys), **kwargs)
        except ImportError as error:
            print("Pandas import error")
            raise error

    def to_xarray(self, **kwargs):
        """ Construct an xarray dataset

        Each column will be converted into an independent variable in the
        Dataset. If the dataframe's index is a MultiIndex, it will be expanded
        into a tensor product of one-dimensional indices (filling in missing
        values with NaN). This method will produce a Dataset very similar to
        that on which the 'to_dataframe' method was called, except with
        possibly redundant dimensions (since all dataset variables will have
        the same dimensionality).

        Parameters
        ----------
        data : ndarray
            (structured dtype), list of tuples, dict, or DataFrame
        keys: sequence, optional
            ordered subset of columns to export
        index : string, list of fields, array-like
            Field of array to use as the index, alternately a specific set of
            input labels to use
        exclude : sequence, default None
            Columns or fields to exclude
        columns : sequence, default None
            Column names to use. If the passed data do not have names
            associated with them, this argument provides names for the
            columns. Otherwise this argument indicates the order of the columns
            in the result (any names not found in the data will become all-NA
            columns)
        coerce_float : boolean, default False
            Attempt to convert values to non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets

        Returns
        -------
        df : DataFrame
        """
        try:
            from xarray import Dataset
            return Dataset.from_dataframe(self.to_pandas(**kwargs))
        except ImportError as error:
            print("xarray import error")
            raise error

    def to_vaex(self, **kwargs):
        """
        Create an in memory Vaex dataset

        Parameters
        ----------
        name: str
            unique for the dataset
        keys: sequence, optional
            ordered subset of columns to export

        Returns
        -------
        df: vaex.DataSetArrays
            vaex dataset
        """
        try:
            import vaex
            return vaex.from_arrays(**self.to_dict(contiguous=True, **kwargs))
        except ImportError as error:
            print("Vaex import error")
            raise error

    def to_dask(self, **kwargs):
        """ Construct a Dask DataFrame

        This splits an in-memory Pandas dataframe into several parts and
        constructs a dask.dataframe from those parts on which Dask.dataframe
        can operate in parallel.

        Note that, despite parallelism, Dask.dataframe may not always be faster
        than Pandas.  We recommend that you stay with Pandas for as long as
        possible before switching to Dask.dataframe.

        Parameters
        ----------
        keys: sequence, optional
            ordered subset of columns to export
        npartitions : int, optional
            The number of partitions of the index to create. Note that
            depending on the size and index of the dataframe, the output may
            have fewer partitions than requested.
        chunksize : int, optional
            The size of the partitions of the index.
        sort: bool
            Sort input first to obtain cleanly divided partitions or don't sort
            and don't get cleanly divided partitions
        name: string, optional
            An optional keyname for the dataframe.  Defaults to hashing the
            input

        Returns
        -------
        dask.DataFrame or dask.Series
            A dask DataFrame/Series partitioned along the index
        """
        try:
            from dask import dataframe
            keys = kwargs.pop('keys', None)
            return dataframe.from_pandas(self.to_pandas(keys=keys), **kwargs)
        except ImportError as error:
            print("Dask import error")
            raise error

    def to_astropy_table(self, **kwargs):
        """
        A class to represent tables of heterogeneous data.

        `astropy.table.Table` provides a class for heterogeneous tabular data,
        making use of a `numpy` structured array internally to store the data
        values.  A key enhancement provided by the `Table` class is the ability
        to easily modify the structure of the table by adding or removing
        columns, or adding new rows of data.  In addition table and column
        metadata are fully supported.

        Parameters
        ----------
        masked : bool, optional
            Specify whether the table is masked.
        names : list, optional
            Specify column names
        dtype : list, optional
            Specify column data types
        meta : dict, optional
            Metadata associated with the table.
        copy : bool, optional
            Copy the input data (default=True).
        rows : numpy ndarray, list of lists, optional
            Row-oriented data for table instead of ``data`` argument
        copy_indices : bool, optional
            Copy any indices in the input data (default=True)
        **kwargs : dict, optional
            Additional keyword args when converting table-like object

        Returns
        -------
        df: astropy.table.Table
            dataframe
        """
        try:
            from astropy.table import Table
            keys = kwargs.pop('keys', None)
            return Table(self.to_records(keys=keys), **kwargs)
        except ImportError as e:
            print("Astropy import error")
            raise e

    def _repr_html_(self):
        return self.to_pandas().head()._repr_html_()

    @property
    def nrows(self):
        """ Number of rows in the dataset """
        return len(self)

    @property
    def ncols(self):
        """ Number of columns in the dataset """
        return dict.__len__(self)

    @classmethod
    def from_lines(cls, iterable):
        """ Create a DataFrame object from its lines instead of columns

        Parameters
        ----------
        iterable: iterable
            sequence of lines with the same keys (expecting dict like
            structure)

        Returns
        -------
        df: DataFrame
            a new object
        """
        d = {}
        default_n = 0
        for line in iterable:
            for k in line.keys():
                d.setdefault(k, [np.atleast_1d(np.nan)] * default_n)\
                 .append(np.atleast_1d(line[k]))
        for k, v in dict.items(d):
            dict.__setitem__(d, k, np.squeeze(np.vstack(v)))

        return cls(d)

    def __repr__(self):
        txt = 'DataFrame ({0:s})\n'.format(pretty_size_print(self.nbytes))
        try:
            txt += '\n'.join([str((k, v.dtype, v.shape))
                              for (k, v) in self.items()])
        except AttributeError:
            txt += '\n'.join([str((k, type(v))) for (k, v) in self.items()])
        return txt

    @property
    def nbytes(self):
        """ number of bytes of the object """
        nbytes = sum(k.nbytes if hasattr(k, 'nbytes') else sys.getsizeof(k)
                     for k in self.__dict__.values())
        return nbytes

    def __getitem__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            return self.evalexpr(k)
        except Exception:
            return self.__class__({a: v[k] for a, v in self.items()})

    @property
    def dtype(self):
        """ the dtypes of each column of the dataset """
        return dict((k, v.dtype) for (k, v) in self.items())

    @property
    def shape(self):
        """ dict of shapes """
        return dict((k, v.shape) for (k, v) in self.items())

    def groupby(self, key):
        """ create an iterator which returns (key, DataFrame) grouped by each
        value of key(value) """
        for k, index in self.arg_groupby(key):
            d = {a: b[index] for a, b in self.items()}
            yield k, self.__class__(d)

    def arg_groupby(self, key):
        """ create an iterator which returns (key, index) grouped by each
        value of key(value) """
        val = self.evalexpr(key)
        ind = sorted(zip(val, range(len(val))), key=lambda x: x[0])

        for k, grp in itertools.groupby(ind, lambda x: x[0]):
            index = [k[1] for k in grp]
            yield k, index

    def __iter__(self):
        """ Iterator on the lines of the dataframe """
        return self.iterlines()

    def iterlines(self):
        """ Iterator on the lines of the dataframe """
        return self.lines

    @property
    def lines(self):
        """ Iterator on the lines of the dataframe """
        for k in range(self.nrows):
            yield self[k]

    @property
    def rows(self):
        """ Iterator on the lines of the dataframe """
        return self.lines

    @property
    def columns(self):
        """ Iterator on the columns
        refers to :func:`dict.items`
        """
        return dict.items(self)

    def where(self, condition, condvars=None):
        """ Read table data fulfilling the given `condition`.
        Only the rows fulfilling the `condition` are included in the result.

        Parameters
        ----------
        query: generator
            generator of records from a query

        condition : str
            The evaluation of this condition should return True or False the
            condition can use field names and variables defined in condvars

        condvars: dict
            dictionary of variables necessary to evaluate the condition.

        Returns
        -------
        it: generator
            iterator on the query content. Each iteration contains one selected
            entry.

        ..note:
            there is no prior check on the variables and names
        """
        for line in self.lines:
            if eval(condition, dict(line), condvars):
                yield line

    def sortby(self, key, reverse=False, copy=False):
        """
        Parameters
        ----------
        key: str
            key to sort on. Must be in the data

        reverse: bool
            if set sort by decreasing order

        copy: bool
            if set returns a new dataframe

        Returns
        -------
        it: DataFrame or None
            new dataframe only if copy is False
        """
        val = self.evalexpr(key)
        ind = np.argsort(val)
        if reverse:
            ind = ind[::-1]
        if not copy:
            for k in self.keys():
                dict.__setitem__(self, k, dict.__getitem__(self, k)[ind])
        else:
            d = {}
            for k in self.keys():
                d[k] = dict.__getitem__(self, k)[ind]
            return self.__class__(d)

    def select(self, keys, caseless=False):
        """ Read table data but returns only selected fields.

        Parameters
        ----------
        keys: str, sequence of str
            field names to select.
            Can be a single field names as follow:
            'RA', or ['RA', 'DEC'], or 'RA,DEC', or 'RA DEC'

        caseless: bool
            if set, do not pay attention to case.

        Returns
        -------
        df: DataFrame
            reduced dataframe

        ..note:
            there is no prior check on the variables and names
        """
        if keys == '*':
            return self

        if caseless:
            _keys = ''.join([k.lower() for k in keys])
            df = self.__class__(dict((k, v)
                                     for k, v in self.items()
                                     if (k.lower() in _keys)))
        else:
            df = self.__class__(dict((k, v)
                                     for k, v in self.items()
                                     if k in keys))
        return df

    def pickle_dump(self, fname):
        """ create a pickle dump of the dataset """
        import pickle
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def unpickle(cls, fname):
        """ restore a previously pickled object """
        import pickle
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def multigroupby(self, key, *args):
        """
        Returns nested grouped DataFrames given the multiple keys

        Parameters
        ----------
        key1, key2, ...: sequence
            keys over which indexing the data

        Returns
        -------
        it: generator
            (key1, (key2, (... keyn, {})))
        """
        return _df_multigroupby(self, key, *args)

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
        pv = [(k, list(v)) for k, v in self.multigroupby(*keys)]
        return _df_multigroupby_aggregate(pv, func=func)

    @property
    def Plotter(self):
        """ Plotter instance related to this dataset.
        Requires plotter add-on to work """
        if Plotter is None:
            msg = 'the add-on was not found, this property is not available'
            raise AttributeError(msg)
        else:
            return Plotter(self)

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
        return evalexpr(self, expr, exprvars=exprvars, dtype=dtype)

    # for common interfaces
    evaluate = evalexpr

    def write(self, fname, keys=None, header=None, **kwargs):
        """Write DictDataFrame into file

        Parameters
        ---------
        fname: string
            file to write into

        keys: sequence, optional
            ordered sequence of columns

        header: string
            header to add to the file

        extension: str
            csv or txt/dat file
            auto guess from fname
            sets the delimiter and comments accordingly

        delimiter: str
            force  the column delimiter

        comments: str
            force the char defining comment line

        kwargs: dict
            forwarded to np.savetxt
        """
        extension = kwargs.pop('extension', None)
        if extension is None:
            extension = fname.split('.')[-1]
        if extension == 'csv':
            comments = kwargs.pop('comments', '#')
            delimiter = kwargs.pop('delimiter', ',')
        elif extension in ('txt', 'dat'):
            comments = kwargs.pop('comments', '#')
            delimiter = kwargs.pop('delimiter', ' ')

        data = self.to_records(keys)
        hdr = comments + " " + delimiter.join(data.dtype.names)

        dtypes = (self.dtype[k] for k in keys)
        fmt = delimiter.join(['%' + k.kind.lower() for k in dtypes])

        # Monkey patch to help unicode/bytes/str mess
        if PY3:
            fmt = fmt.replace('u', 's')

        np.savetxt(fname, data, delimiter=delimiter,
                   header=header + hdr, fmt=fmt,
                   comments='', **kwargs)

    def join(self, key, other, key_other=None, columns_other=None,
             prefix=None):
        """
        Experimental joining of structures, (equivalent to SQL left outer join)
        Example:
        >>> x = np.arange(10)
        >>> y = x**2
        >>> z = x**3
        >>> df = DictDataFrame(x=x, y=y)
        >>> df2 = DictDataFrame(x=x[:4], z=z[:4])
        >>> df.join('x', df2, 'x', column_other=['z'])

        Parameters
        ----------
        key: str
            key for the left table (self)

        other: DictDataFrame
            Other dataset to join with (the right side of the outer join)

        key_other: str, optional
            key on which to join (default identical to key)

        columns_other: tuple, optional
            column names to add to the dataframe (default: all fields)
        prefix: str, optional
            add a prefix to the new column

        Returns
        -------
        self: DictDataFrame
            itself
        """
        N = len(self)
        N_other = len(other)
        if columns_other is None:
            columns_other = list(other.keys())
        if key_other is None:
            key_other = key

        # Bullet proofing existing data
        if prefix is None:
            for name in columns_other:
                if name in self:
                    msg = "Field {0:s} already exists.".format(name)
                    raise ValueError(msg)
            else:
                for name in columns_other:
                    new_name = '{0:s}{1:s}'.format(prefix, name)
                    msg = "Field {0:s} already exists.".format(new_name)
                    if new_name in self:
                        raise ValueError(msg)

        # generate index
        key = self[key]
        key_other = other[key_other]
        index = dict(zip(key, range(N)))
        index_other = dict(zip(key_other, range(N_other)))

        from_indices = np.zeros(N_other, dtype=np.int64)
        to_indices = np.zeros(N_other, dtype=np.int64)
        for i in range(N_other):
            if key_other[i] in index:
                to_indices[i] = index[key_other[i]]
                from_indices[i] = index_other[key_other[i]]
            else:
                to_indices[i] = -1
                from_indices[i] = -1

        mask = to_indices != -1
        to_indices = to_indices[mask]
        from_indices = from_indices[mask]

        dtypes = other.dtype
        for column_name in columns_other:
            dtype = dtypes[column_name]
            if np.issubdtype(dtype, np.inexact):
                data = np.zeros(N, dtype=dtype)
                data[:] = np.nan
                data[to_indices] = other[column_name][from_indices]
            else:
                data = np.ma.masked_all(N, dtype=dtype)
                values = other[column_name][from_indices]
                data[to_indices] = values
                data.mask[to_indices] = np.ma.masked
                if not np.ma.is_masked(data):
                    # forget the mask if we do not need it
                    data = data.data
            if prefix:
                new_name = '{0:s}{1:s}'.format(prefix, column_name)
            else:
                new_name = column_name
            self[new_name] = data
        return self


def _convert_dict_to_structured_ndarray(data, keys=None):
    """ convert_dict_to_structured_ndarray

    Parameters
    ----------
    data: dictionary like object
        data structure which provides iteritems and itervalues
    keys: sequence, optional
        ordered subset of columns to export

    returns
    -------
    tab: structured ndarray
        structured numpy array
    """
    newdtype = []
    if keys is None:
        keys = data.keys()
    for key in keys:
        _dk = data[key]
        dtype = data.dtype[key]
        # unknown type is converted to text
        if dtype.type == np.object_:
            if len(data) == 0:
                longest = 0
            else:
                longest = len(max(_dk, key=len))
                _dk = _dk.astype('|%iS' % longest)
        if _dk.ndim > 1:
            newdtype.append((str(key), _dk.dtype, (_dk.shape[1],)))
        else:
            newdtype.append((str(key), _dk.dtype))
    tab = np.rec.fromarrays((data[k] for k in keys), dtype=newdtype)
    return tab


def _df_multigroupby(ary, *args):
    """
    Generate nested df based on multiple grouping keys

    Parameters
    ----------
    ary: dataFrame, dict like structure

    args: str or sequence
        column(s) to index the DF
    """
    if len(args) <= 0:
        yield ary
    elif len(args) > 1:
        nested = True
    else:
        nested = False

    val = ary[args[0]]
    ind = sorted(zip(val, range(len(val))), key=lambda x: x[0])

    for k, grp in itertools.groupby(ind, lambda x: x[0]):
        index = [v[1] for v in grp]
        d = ary.__class__({a: np.array([b[i] for i in index])
                           for a, b in ary.items()})
        if nested:
            yield k, _df_multigroupby(d, *args[1:])
        else:
            yield k, d


def _df_multigroupby_aggregate(pv, func=lambda x: x):
    """
    Generate a flattened structure from multigroupby result

    Parameters
    ----------
    pv: dataFrame, dict like structure
    result from :func:`_df_multigroupby`

    func: callable
        reduce the data according to this function (default: identity)

    Returns
    -------
    seq: sequence
        flattened sequence of keys and value
    """
    def aggregate(a, b=()):
        data = []
        for k, v in a:
            if type(v) in (list, tuple,):
                data.append(aggregate(v, b=(k,)))
            else:
                data.append(b + (k, func(v)))
        return data
    return list(itertools.chain(*aggregate(pv)))


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
    if expr in keys:
        return data[expr]
    for k in keys:
        if k in expr:
            _globals[k] = data[k]

    if exprvars is not None:
        if not hasattr(exprvars, 'items'):
            msg = "Expecting a dict-like as condvars with an `items` method"
            raise AttributeError(msg)
        for k, v in (exprvars.items()):
            _globals[k] = v

    # evaluate expression, to obtain the final filter
    # r = np.empty( self.nrows, dtype=dtype)
    r = eval(expr, _globals, np.__dict__)

    return np.array(r, dtype=dtype)
