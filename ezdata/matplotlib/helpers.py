"""
Some mpl helpers
"""
import pylab as plt


def is_colorbar(ax):
    """ Guesses whether a set of Axes is hosting a colorbar

    Assumptions: a colorbar has a data aspect ratio of 1. and is not navigable.

    Parameters
    ----------

    ax: Axes instance
        axes to test

    Returns
    -------
    is_colorbar: bool
        True if it guesses it's a colorbar
    """
    return (ax.get_data_ratio() == 1.0 and not ax.get_navigate())


def label_subplots(axes=None, fmt='{0:d}', uppercase=True, **kwargs):
    """ Add a letter label to each axes """
    import string
    if axes is None:
        axes = [ax for ax in plt.gcf().get_axes() if not is_colorbar(ax)]
    if not uppercase:
        labels = string.ascii_lowercase
    else:
        labels = string.ascii_uppercase

    defaults = dict(fontsize='large', fontweight='bold',
                    va='center', ha='right')
    defaults.update(kwargs)

    for label, ax in zip(labels, axes):
        ax.text(-0.08, 1., "{0}".format(label),
                transform=ax.transAxes, **defaults)
