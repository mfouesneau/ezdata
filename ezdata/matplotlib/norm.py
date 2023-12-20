""" MPL normalization functions """
import numpy as np
from matplotlib import colors


class DummyNorm(colors.Normalize):
    """ A template to generate a new Matplotlib Norm transformation

    Basically you only need to derive this class and set the
    `_transform` and `_inverse_transform` methods.
    """
    def _transform(self, value, out=None):
        delta = self.vmax - self.vmin
        return (value - self.vmin) / float(delta)

    def _inverse_transform(self, value, out=None):
        delta = self.vmax - self.vmin
        return delta * value + self.vmin

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        # Normalize initial recipe to clean inputs
        if clip:
            result, is_scalar = self.process_value(np.clip(value),
                                                   self.vmin,
                                                   self.vmax)
        else:
            result, is_scalar = self.process_value(value)

        resdat = result.data
        resdat = self._transform(resdat)
        result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            return result[0]
        return result

    def inverse(self, value):
        result, is_scalar = self.process_value(value)
        resdat = result.data
        resdat = self._inverse_transform(resdat)
        result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            return result[0]
        return result


class HistEq(DummyNorm):
    """
    Histogram equalizer normalization

    Attributes
    ----------
    artist: Artist instance or ndarray
        artist or image to normalize
    bins: int, optional
        number of bins to calculate the histogram
    """
    def __init__(self, artist, *args, **kwargs):
        bins = kwargs.pop('bins', 1024)
        super().__init__(*args, **kwargs)
        try:
            image = artist._A
        except AttributeError:
            image = artist
        values, bins = np.histogram(image.ravel(), bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        csum = values.cumsum()
        self.histogram = centers, csum / csum[-1], values

    def _transform(self, value, out=None):
        return np.interp(value, self.histogram[0], self.histogram[1])

    def _inverse_transform(self, value, out=None):
        return np.interp(value, self.histogram[1], self.histogram[0])


class Arcsinh(DummyNorm):
    """ Arcsinh normalization """
    def _transform(self, value, out=None):
        delta = np.arcsinh(self.vmax) - np.arcsinh(self.vmin)
        return (np.arcsinh(value) - np.arcsinh(self.vmin)) / float(delta)

    def _inverse_transform(self, value, out=None):
        delta = np.arcsinh(self.vmax) - np.arcsinh(self.vmin)
        return np.sinh(delta * value + np.arcsinh(self.vmin))
    
    
class MidpointNormalize(colors.Normalize):
    """ Normalize to a central point """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

    
class Sqrt(DummyNorm):
    """ Sqrt normalization """
    def _transform(self, value, out=None):
        delta = np.sqrt(self.vmax) - np.sqrt(self.vmin)
        return (np.sqrt(value) - np.sqrt(self.vmin)) / float(delta)

    def _inverse_transform(self, value, out=None):
        delta = np.power(self.vmax, 2) - np.power(self.vmin, 2)
        return np.power(delta * value + np.power(self.vmin, 2))

    
class Power(DummyNorm):
    """ Power normalization """
    def __init__(self, power_value=2, vmin=None, vmax=None, clip=False):
        self.power_ = power_value
        self.inv_power_ = 1. / power_value
        colors.Normalize.__init__(self, vmin, vmax, clip)
        
    def _transform(self, value, out=None):
        delta = np.power(self.vmax, self.power_) - np.power(self.vmin, self.power_)
        return (np.power(value, self.power_) - np.power(self.vmin, self.power_)) / float(delta)

    def _inverse_transform(self, value, out=None):
        delta = np.power(self.vmax, self.inv_power_) - np.power(self.vmin, self.inv_power_)
        return np.power(delta * value + np.power(self.vmin, self.inv_power_))


class PercentileNormalize(DummyNorm):
    """ Using vmin, vmax percentiles as normalization. """ 
    def __init__(self, data, vmin=None, vmax=None, clip=False):
        vmin, vmax = np.nanpercentile(data, [vmin, vmax])
        super().__init__(vmin, vmax, clip)
        self.data = data

class StdNormalize(DummyNorm):
    """ Using mean and standard deviation as normalization 
    use vmin, vmax to clip the range and nstd to set how many standard deviations to consider 
    """ 
    def __init__(self, data, vmin=None, vmax=None, clip=False, nstd=3.0):
        self.data = data
        mean = np.nanmean(self.data)
        std = np.nanstd(self.data)
        a, b = mean - nstd * std, mean + nstd * std 
        vmin = max(a, vmin or a)
        vmax = min(b, vmax or b)
        super().__init__(vmin, vmax, clip)
