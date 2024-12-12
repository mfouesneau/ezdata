"""
Vaex has become less maintained these days, and its hdf structure is non-standard.
This script allows to quickly convert a vaex hdf5 file to a parquet file

It requires h5py and pyarrow.
"""
from typing import Sequence

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:
    pass


def vaex_to_parquet(
    hdf5_file: str,
    parquet_file: str,
    grouppath: str = "/table/columns",
    chunk_size: int = 1_000,
    columns: Sequence[str] = None,
    progress: bool = False,
) -> str:
    """Convert a vaex formatted hdf5 file to a parquet file

    Parameters
    ----------
    hdf5_file : str
        path to the hdf5 file
    parquet_file : str
        path to the parquet file
    grouppath : str
        root path to the data, vaex default is '/table/columns'
    chunk_size : int
        chunk size to read in memory at once, default is 1_000
    columns : Sequence[str]
        list of columns to export, default is all
    progress : bool
        whether to show a progress bar

    Returns
    -------
    str
        path to the output parquet file

    Raises
    ------
    ValueError:
        if all fields do not have the same shape.
        This converter assumes the data are in a flat table (1d-fields).

    Note
    ----
    - This converter assumes the data are in a flat table (1d-fields).
    - Vaex file format is essentially one array per field:/table/columns/<field>/data
    - This version does not deal with mutliple files.
    """
    # set progress indicator if needed.
    if not progress:
        _progress = lambda x, **kwargs: x  # noqa
    else:
        _progress = tqdm

    def _read_array_slice(
        hdf: h5py.File, grouppath: str, field: str, start: int, end: int
    ) -> np.ndarray:
        """Read a slice of an array from the HDF5 file"""
        data = hdf[grouppath][field]
        end = min(end, data.shape[0])
        return data[start:end]

    with h5py.File(hdf5_file, "r") as hdf:
        # Get the list of fields (columns)
        if columns is None:
            names = list(hdf[grouppath].keys())
            fields = [f"{name}/data" for name in names]
        else:
            fields = columns
            names = [field.replace("/data", "") for field in fields]

        # Get the list of data types and shapes
        dtypes = [hdf[grouppath][field].dtype for field in fields]
        shapes = [hdf[grouppath][field].shape for field in fields]

        # check shapes
        if len(set(shapes)) > 1:
            raise ValueError(
                f"All fields must have the same shape, found {set(shapes)}"
            )

        # Build the schema of the output table (names, dtypes)
        schema = pa.schema(
            [
                pa.field(name, pa.from_numpy_dtype(dtype))
                for field, name, dtype in zip(fields, names, dtypes)
            ]
        )

        # Create a Parquet writer
        num_rows = hdf[grouppath][fields[0]].shape[0]
        with pq.ParquetWriter(parquet_file, schema) as writer:
            for start in _progress(
                range(0, num_rows, chunk_size), total=num_rows // chunk_size
            ):
                end = start + chunk_size
                arrays = [
                    _read_array_slice(hdf, grouppath, field, start, end)
                    for field in fields
                ]
                table = pa.Table.from_arrays(
                    [pa.array(array) for array in arrays], names=names
                )
                # Write the chunk to the Parquet file
                writer.write_table(table)

    return parquet_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a vaex hdf5 file to parquet")
    parser.add_argument("hdf5_file", type=str, help="path to the hdf5 file")
    parser.add_argument("parquet_file", type=str, help="path to the parquet file")
    parser.add_argument(
        "-g",
        "--grouppath",
        type=str,
        default="/table/columns",
        help="root path to the data, default is '/table/columns'",
    )
    parser.add_argument(
        "-n",
        "--chunk_size",
        type=int,
        default=1_000,
        help="chunk size to read in memory at once, default is 1_000",
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=str,
        nargs="+",
        default=None,
        help="list of columns to export, default is all",
    )
    parser.add_argument(
        "-p",
        "--progress",
        action="store_true",
        help="whether to show a progress bar",
    )

    args = parser.parse_args()
    vaex_to_parquet(
        args.hdf5_file,
        args.parquet_file,
        args.grouppath,
        args.chunk_size,
        args.columns,
        args.progress,
    )
