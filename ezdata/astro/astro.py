""" Astronomy related functions """
from functools import wraps
import numpy as np

try:
    import dask
except ImportError:
    dask = None


def dask_compatibility(fn):
    """ Make functions transparent to using dask delayed objects """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as issue:
            if dask is None:
                raise issue
            return dask.delayed(fn)(*args, **kwargs).compute()

    return wrapped
 

def nside2npix(nside):
    """Give the number of pixels for the given nside.

    Parameters
    ----------
    nside : int
      healpix nside parameter; an exception is raised if nside is not valid
      (nside must be a power of 2, less than 2**30)

    Returns
    -------
    npix : int
      corresponding number of pixels

    Notes
    -----
    Raise a ValueError exception if nside is not valid.

    Examples
    --------
    >>> import numpy as np
    >>> nside2npix(8)
    768
    """
    return 12 * nside * nside


def gaia_healpix_expression(healpix_expression="source_id/34359738368",
                            healpix_max_level=12, healpix_level=8):
    """
    Give the healpix expression from the Gaia source_id at
    a given healpix level

    Parameters
    ----------
    healpix_expression: str
        field name and conversion to healpix cell
    healpix_max_level: int
        expression corresponding level
    healpix_level: int
        desired healpix level from the data

    Returns
    -------
    expression: str
        final expression
    """
    reduce_level = healpix_max_level - healpix_level
    # NSIDE = 2 ** healpix_level
    # nmax = nside2npix(NSIDE)
    scaling = 4 ** reduce_level
    # epsilon = 1. / scaling / 2
    expression = "%s/%s" % (healpix_expression, scaling)
    return expression


@dask_compatibility
def get_healpix_grid(data, healpix_level):
    """Convert a dataframe to the dense grid

    Parameters
    ----------
    data: pd.DataFrame
        data from a database query

    healpix_level: int
        level of the query

    Returns
    -------
    grid: np.array (npix, )
        dense grid of npix(healpix_level) with the data values
    """
    grid = np.zeros(nside2npix(2 ** healpix_level), dtype=data.n.values.dtype)
    grid[data.hpx] = data.n
    return grid


def healpix_grid_plot(fgrid, what_label=None, colormap="afmhot",
                      grid_limits=None, healpix_input="equatorial",
                      healpix_output="galactic", image_size=800, nest=True,
                      norm=None, title="", smooth=None,
                      colorbar=True, rotation=(0, 0, 0), **kwargs):
    """ Plot data from healpix configuration
    what_label: str
        colorbar label
    colormap: str or cmap instance
        colormap used by matplotlib
    healpix_input: str
        Specificy if the healpix index is in
        "equatorial", "galactic" or "ecliptic".
    healpix_output: str
        Plot in "equatorial", "galactic" or "ecliptic".
    grid_limits: tuple, optional
        [minvalue, maxvalue] value that map to the colormap
        (values below and above these are clipped to the the min/max).
    image_size: int
        size for the image that healpy uses for rendering
    nest: boolean
        If the healpix data is in nested (True) or ring (False)
    title: str
        Title of figure
    smooth: float
        apply gaussian smoothing, in degrees
    rotation: tuple(3)
        Rotate the plot, in format (lon, lat, psi)
        such that (lon, lat) is the center,
        and rotate on the screen by angle psi. All angles are degrees.
    norm : {'hist', 'log', None}
      Color normalization, hist= histogram equalized color mapping,
      log= logarithmic color mapping, default: None (linear color mapping)
    """
    import healpy as hp
    from matplotlib import colors
    import warnings

    # Compatibility filter
    colormap = kwargs.pop('cmap', colormap)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    if grid_limits is None:
        grid_limits = [vmin, vmax]
    if isinstance(norm, colors.LogNorm):
        norm = 'log'
        if grid_limits[0] is None:
            grid_limits[0] = 1

    if smooth:
        if nest:
            grid = hp.reorder(fgrid, inp="NEST", out="RING")
            nest = False
        # grid[np.isnan(grid)] = np.nanmean(grid)
        grid = hp.smoothing(fgrid, sigma=np.radians(smooth))
    else:
        grid = fgrid
    if grid_limits:
        grid_min, grid_max = grid_limits
    else:
        grid_min = grid_max = None
    func = hp.mollview
    coord_map = dict(equatorial='C', galactic='G', ecliptic="E")
    coord = coord_map[healpix_input], coord_map[healpix_output]
    if coord_map[healpix_input] == coord_map[healpix_output]:
        coord = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        func(grid, unit=what_label, rot=rotation, nest=nest,
             title=title, coord=coord, cmap=colormap, hold=True,
             xsize=image_size, min=grid_min, norm=norm,
             max=grid_max, cbar=colorbar, **kwargs)


@dask_compatibility
def add_column_healpix(self, name="healpix", longitude="ra", latitude="dec",
                       degrees=True, healpix_order=12, nest=True):
    """Add a healpix (in memory) column based on a longitude and latitude

    Parameters
    ----------
    name: str
        name of the column
    longitude: str
        expression of the longitude (or right-ascension) coordinate
        (astronomical convenction latitude=90 is north pole)
    latitude: str
        expression of the latitude (or declinasion) coordinate
    degrees: boolean
        If lon/lat are in degrees (default) or radians.
    healpix_order: int
        healpix order, >= 0
    nest: boolean
        Nested healpix (default) or ring.
    """
    import healpy as hp
    if degrees:
        scale = np.pi / 180.
    else:
        scale = 1.

    phi = self[longitude] * scale
    theta = np.pi / 2 - self[latitude] * scale
    hp_index = hp.ang2pix(hp.order2nside(healpix_order), theta, phi, nest=nest)
    try:
        self.add_column("healpix", hp_index)
    except AttributeError:
        self['healpix'] = hp_index
    return self


@dask_compatibility
def project_aitoff(alphain, deltain, radians=True):
    """Add aitoff (https://en.wikipedia.org/wiki/Aitoff_projection) projection

    TODO: optimize for DASK DataFrame

    Parameters
    ----------
    alpha: array
        azimuth angle
    delta: array
        polar angle
    radians: boolean
        input and output in radians (True), or degrees (False)

    returns
    -------
    x: ndarray
        x coordinate
    y: ndarray
        y coordinate
    """
    try:
        transform = 1. if radians else np.pi / 180.
        alpha = np.copy(alphain)
        if not radians:
            ind = alphain > 180
            alpha[ind] = alphain[ind] - 360
        else:
            ind = alphain > np.pi
            alpha[ind] = alphain[ind] - 2. * np.pi
        delta = deltain

        aitoff_alpha = np.arccos(np.cos(delta * transform) *
                                 np.cos(0.5 * alpha * transform))
        x = (2 * np.cos(delta * transform) * np.sin(0.5 * alpha * transform) /
             np.sinc(aitoff_alpha / np.pi) / np.pi)
        y = np.sin(delta * transform) / np.sinc(aitoff_alpha / np.pi) / np.pi

        return x, y
    except ValueError as issue:
        # dask df are not playing nice with the above
        try:
            import dask
            return dask.delayed(project_aitoff)(alphain, deltain, radians)\
                       .compute()
        except ImportError:
            raise issue


def add_aitoff_projections(self, alpha, delta, x, y, radians=False):
    """Add aitoff (https://en.wikipedia.org/wiki/Aitoff_projection) projection

    Parameters
    ----------
    alpha: array
        azimuth angle
    delta: array
        polar angle
    radians: boolean
        input and output in radians (True), or degrees (False)
    x: str
        output name for x coordinate
    y: str
        output name for y coordinate

    returns
    -------
    x: ndarray
        output name for x coordinate
    y: ndarray
        output name for y coordinate
    """
    x_, y_ = project_aitoff(self[alpha], self[delta], radians=radians)
    try:
        self.add_column(x, x_)
    except AttributeError:
        self[x] = x_
    try:
        self.add_column(y, y_)
    except AttributeError:
        self[y] = y_
    return self


def find_matching_parenthesis(string):
    """ Find recursively groups of balanced parenthesis """
    stack = 0
    startIndex = None
    results = []

    for i, c in enumerate(string):
        if c == '(':
            if stack == 0:
                startIndex = i + 1  # string to extract starts one index later

            # push to stack
            stack += 1
        elif c == ')':
            # pop stack
            stack -= 1

            if stack == 0:
                results.append(string[startIndex:i])

    rprime = [find_matching_parenthesis(rk) for rk in results if len(results)]

    if len(results):
        if len(rprime):
            return results + rprime
        else:
            return results


def flatten(lst):
    """ Flatten a nest list or nested sequence of values """
    res = []
    for k in lst:
        if isinstance(k, (list, tuple)):
            res.extend(flatten(k))
        else:
            if k is not None:
                res.append(k)
    return res


def healpix_plot(self, healpix_expression='healpix', healpix_level=8,
                 what='count(*)', grid=None,
                 healpix_input='equatorial', healpix_output='galactic',
                 norm=None, colormap='afmhot', grid_limits=None,
                 image_size=800, nest=True,
                 title='', smooth=None, colorbar=True,
                 rotation=(0, 0, 0), **kwargs):
    """ Plot data from healpix configuration
    what_label: str
        colorbar label
    colormap: str or cmap instance
        colormap used by matplotlib
    grid: ndarray
        healpix grid of size nside2npix(2 ** level)
    healpix_input: str
        Specificy if the healpix index is in
        "equatorial", "galactic" or "ecliptic".
    healpix_output: str
        Plot in "equatorial", "galactic" or "ecliptic".
    grid_limits: tuple, optional
        [minvalue, maxvalue] value that map to the colormap
        (values below and above these are clipped to the the min/max).
    image_size: int
        size for the image that healpy uses for rendering
    nest: boolean
        If the healpix data is in nested (True) or ring (False)
    title: str
        Title of figure
    smooth: float
        apply gaussian smoothing, in degrees
    rotation: tuple(3)
        Rotate the plot, in format (lon, lat, psi)
        such that (lon, lat) is the center,
        and rotate on the screen by angle psi. All angles are degrees.
    norm : {'hist', 'log', None}
      Color normalization, hist= histogram equalized color mapping,
      log= logarithmic color mapping, default: None (linear color mapping)
    """
    from scipy.stats import binned_statistic

    if grid is None:
        try:
            what_ = find_matching_parenthesis(what)[0]
        except TypeError:
            what_ = what
        func = what.replace(what_, '')[:-2]  # remove ()
        if what_ in ('*', ):
            value = self[healpix_expression]
        else:
            value = self[what_]
        binned_statistic_ = dask_compatibility(binned_statistic)
        grid = binned_statistic_(self[healpix_expression],
                                 value,
                                 bins=nside2npix(2 ** healpix_level),
                                 statistic=func).statistic

    return healpix_grid_plot(grid, what_label=what, colormap=colormap,
                             grid_limits=grid_limits,
                             healpix_input=healpix_input,
                             healpix_output=healpix_output,
                             image_size=image_size, nest=nest, norm=norm,
                             title=title, smooth=smooth,
                             colorbar=colorbar, rotation=rotation, **kwargs)
