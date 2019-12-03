""" Bokeh theme equivalent to the matplotlib theme

>>> import plotly.plotly as py
    from plotly import tools
    import plotly.graph_objs as go
    # stack two subplots vertically
    fig = tools.get_subplots(rows=2)
    fig.append_trace(
            Scatter(x=[1,2,3], y=[2,1,2], xaxis='x1', yaxis='y1'), 1, 1)
    fig.append_trace(
            Scatter(x=[1,2,3], y=[2,1,2], xaxis='x2', yaxis='y2'), 2, 1)
    fig['layout'].update(get_theme())

>>> import holoviews as hv
    style = get_theme()
    hv.renderer('bokeh').theme = style

>>> import bokeh.plotting
    style = get_theme('14pt')
    bokeh.plotting.curdoc().theme = style
"""
import plotly.graph_objs as go


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
        return int(_get_font_size(which, default_fontsize)[:-2])

    DEFAULT_PLOTLY_LAYOUT = go.Layout(
        hovermode=False,
        height=500,
        width=700,
        titlefont=dict(size=get_font_size('x-large')),
        margin=dict(t=80, r=100),
        showlegend=True,
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            font=dict(
                family='serif',
                size=get_font_size('medium'),
            )),
        xaxis=dict(
            titlefont=dict(size=get_font_size('large')),
            ticklen=5,
            gridwidth=2,
            ticks='outside',
            showline=True,
            showgrid=False,
            zeroline=False,
            tickcolor='#666666',
            linecolor='#666666',
            tickfont=dict(
                family='serif',
                size=get_font_size('small'),
                color='#666666'
            ),
        ),
        yaxis=dict(
            titlefont=dict(size=get_font_size('large')),
            ticklen=5,
            ticks='outside',
            gridwidth=2,
            showline=True,
            showgrid=False,
            zeroline=False,
            tickcolor='#666666',
            linecolor='#666666',
            tickfont=dict(
                family='serif',
                size=get_font_size('small'),
                color='#666666'
            ),
        ))
    return DEFAULT_PLOTLY_LAYOUT
