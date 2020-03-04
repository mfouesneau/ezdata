""" Making dask and pandas easier to interface with ezdata API 

This mostly gives an extension to the __getitem__ method of both 
pandas and dask DataFrame objects which allows for expressions
"""
from __future__ import absolute_import

map_ = {}

try:
    import pandas as pd_

    class DummyPandas(pd_.DataFrame):

        def __init__(self, df, *args, **kwargs):
            try:
                super().__init__(df, *args, **kwargs)
            except Exception:
                self.__dict__.update(df.__dict__)

        def __getitem__(self, key):
            try:
                return pd_.DataFrame.__getitem__(self, key)
            except KeyError:
                pd_.DataFrame.__setitem__(self, key, self.eval(key))
            return pd_.DataFrame.__getitem__(self, key)

    map_[pd_.DataFrame] = DummyPandas

except ImportError:
    pass

try:
    import dask.dataframe as dd_

    class DummyDask(dd_.DataFrame):

        def __init__(self, df, *args, **kwargs):
            try:
                super().__init__(df, *args, **kwargs)
            except Exception:
                self.__dict__.update(df.__dict__)

        def __getitem__(self, key):
            try:
                return dd_.DataFrame.__getitem__(self, key)
            except KeyError:
                dd_.DataFrame.__setitem__(self, key, self.eval(key))
            return dd_.DataFrame.__getitem__(self, key)
        
        def __contains__(self, obj):
            return obj in self.columns 

    map_[dd_.DataFrame] = DummyDask
except ImportError:
    pass


def dummy(dataframe):
    """ Get the wrapped version of the dataframe """
    return map_[dataframe.__class__](dataframe)
