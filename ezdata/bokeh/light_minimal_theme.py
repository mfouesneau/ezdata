""" Bokeh theme equivalent to the matplotlib theme

>>> import holoviews as hv
    style = get_theme()
    hv.renderer('bokeh').theme = style

>>> import bokeh.plotting
    style = get_theme('14pt')
    bokeh.plotting.curdoc().theme = style
"""
from bokeh.themes import Theme


def _get_font_size(which, normal='12pt'):
    """ Generates the equivalent of matplotlib fontsize scaling """
    font_scalings = {
        'xx-small': 0.579,
        'x-small': 0.694,
        'small': 0.833,
        'medium': 1.0,
        'large': 1.200,
        'x-large': 1.440,
        'xx-large': 1.728,
        'larger': 1.2,
        'smaller': 0.833,
        None: 1.0}
    try:
        val = int(float(normal.replace('pt', ''))
                  * font_scalings.get(which.lower(), 1.))
        return '{0:d}pt'.format(val)
    except Exception:
        return which


def get_theme(default_fontsize='12pt'):
    """ Generate the theme for the given font size """

    def get_font_size(which):
        return _get_font_size(which, default_fontsize)

    style = Theme(json={'attrs': {

        # apply defaults to Figure properties
        'Figure': {
            # toolbar_location in ["above" "below" "left" "right"  None]
            'toolbar_location': 'right',
            'outline_line_color': None,
            'min_border_right': 10,
            'width': 400,
            'height': 300,
        },

        # apply defaults to Axis properties
        'Axis': {
            'major_label_text_font_size': get_font_size('small'),
            'major_label_text_font_style': 'normal',
            'major_label_text_color': '#666666',
            "major_tick_line_alpha": 1.,
            "major_label_text_font": "DejaVu Sans",

            "major_tick_line_color": "#666666",
            "minor_tick_line_alpha": 1.,
            "minor_tick_line_color": "#666666",

            "axis_line_alpha": 1.,
            "axis_line_color": "#666666",
            "axis_label_text_font": "DejaVu Sans",
            'axis_label_text_font_style': 'normal',
            "axis_label_text_color": "#000000",
            'axis_label_text_font_size': get_font_size('large'),
            'major_tick_in': None,
            'minor_tick_out': None,
            'minor_tick_in': None,
        },

        # apply defaults to Legend properties
        'Legend': {
            "spacing": 8,
            "glyph_width": 15,

            "label_standoff": 8,
            "label_text_color": "#000000",
            "label_text_font": "DejaVu Sans",
            "label_text_font_size": get_font_size('small'),

            "border_line_alpha": 0,
            "background_fill_alpha": 0.25
        },

        "ColorBar": {
            "title_text_color": "#000000",
            "title_text_font": "DejaVu Sans",
            "title_text_alpha": 1,
            "title_text_font_size": get_font_size('large'),
            "title_text_font_style": "normal",
            "title_text_line_height": 1.2,

            "major_label_text_color": "#666666",
            "major_label_text_font": "DejaVu Sans",
            "major_label_text_font_size": get_font_size('small'),
            'major_tick_out': 8,
            'major_tick_in': None,

            "major_tick_line_alpha": 1.,
            "major_tick_line_color": "#666666",
            "bar_line_alpha": 1,
            "width":  15   # 'auto'
        },

        "Title": {
            "text_color": "#000000",
            "text_font": "DejaVu Sans",
            "text_font_size": get_font_size('x-large')
        }
    }})
    return style
