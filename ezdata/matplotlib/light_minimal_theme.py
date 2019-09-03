""" Minimal theme for publication quality with matplotlib.

>>> import pylab as plt
    plt.style.use(light_minimal)
"""

light_minimal = {
    'font.family': 'serif',
    'font.size': 14,
    "axes.titlesize": "x-large",
    "axes.labelsize": "large",
    'axes.edgecolor': '#666666',
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": "8",
    "xtick.minor.size": "4",
    "ytick.major.size": "8",
    "ytick.minor.size": "4",
    'xtick.labelsize': 'small',
    'ytick.labelsize': 'small',
    'ytick.color': '#666666',
    'xtick.color': '#666666',
    'xtick.top': False,
    'ytick.right': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'image.aspect': 'auto'
}


def use(*args, **kwargs):
    """ Set theme as default """
    import pylab as plt
    plt.style.use(light_minimal)
