"""
Matplotlib does not handle images in its legends.
This package updates this issue by offering a legend function 
that can deal with adding a colorbar symbol in the normal legend (or apart)
to indicate objects.

This applies to histograms, images, and other artists.
"""
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
import matplotlib.legend as mlegend
from matplotlib.legend import Legend


class HandlerColormap(HandlerBase):
    def __init__(self, cmap, num_stripes=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes
        
    def create_artists(self, legend, orig_handle, 
                       xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([xdescent + i * width / self.num_stripes, ydescent], 
                          width / self.num_stripes, 
                          height, 
                          fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)), 
                          transform=trans)
            stripes.append(s)
        return stripes

    
def add_image_legend(artists=None, labels=None, ax=None, 
                     merge=True, **kwargs):
    """
    Generate the legend associated with artists of the current/given axes.
    This may be used to create the complete legend or keep artists separated.
    
    Call signatures::

    add_image_legend()
    add_image_legend(labels)
    add_image_legend(artists, labels)
    
    see also: `plt.legend`
    
    Parameters
    ----------
    artists : sequence of `.Artist`, optional
        A list of Artists (lines, patches, images) to be added to the legend.
        Use this together with *labels*, if you need full control on what
        is shown in the legend and the automatic mechanism described above
        is not sufficient.

        The length of handles and labels should be the same in this
        case. If they are not, they are truncated to the smaller length.

    labels : sequence of strings, optional
        A list of labels to show next to the artists.
        Use this together with *handles*, if you need full control on what
        is shown in the legend and the automatic mechanism described above
        is not sufficient or if labels were not given to the artists.
        
    ax: matplotlib.Axes, optional
        legend will be added to this axes (or currently active axes)
    
    merge: boolean, optional
      set to merge this legend with the normal legend (lines, dots, ...)
      If set the full legend will be setup. 
    
    Other arguments are forwarded to `Axes.legend()`
    """
    if ax is None:
        ax = plt.gca()

    if artists is None:
        artists = ax.artists
        
    # take artists only if they have a label
    if (labels is not None):
        if len(labels) == len(artists):
            labels_ = labels
            artists_ = artists
    else:
        labels_ = []
        artists_ = []
        
        for k, artk in enumerate(artists):
            try:
                lbl = artk.get_label()
                if lbl[0] == '_':
                    # not to be in legends
                    raise AttributeError("hidden item")
                labels_.append(lbl)
                artists_.append(artk)
            except AttributeError as e:
                pass
        
    
    # get their colormaps
    cmaps = [pk.get_cmap() for pk in artists_]
    # create proxy artists as handles:
    handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
    handler_map = dict(zip(handles, 
                           [HandlerColormap(cm, num_stripes=8) for cm in cmaps]))
    if merge:
        # get the other legend items
        other_handles, other_labels = ax.get_legend_handles_labels()
        handles.extend(other_handles)
        default_handler_map = mlegend.Legend.get_default_handler_map()
        for new_handle, new_label in zip(other_handles, other_labels):
            hd = mlegend.Legend.get_legend_handler(default_handler_map, 
                                                   new_handle)
            handler_map[new_handle] = hd
            labels_.append(new_label)
     
        return plt.legend(handles=handles, 
                          labels=labels_, 
                          handler_map=handler_map, 
                          **kwargs)
    else:
        leg = Legend(ax, 
                     handles=handles, 
                     labels=labels_, 
                     handler_map=handler_map,
                     **kwargs)
        ax.add_artist(leg);
        return leg
