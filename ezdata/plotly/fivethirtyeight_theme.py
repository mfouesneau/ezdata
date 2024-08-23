""" Define a fivethirtyeight template for ploty

from ezdata.plotly import fivethirtyeight
import plotly.express as px
px.plot(range(10), template='fivethirtyeight')
"""
import plotly.graph_objects as go
import plotly.io as pio

# Define the template
template = go.layout.Template()

# Line settings
template.data.scatter = [{
    'line': {'width': 4}
}]

# Legend settings
template.layout.legend = dict(
    bgcolor='rgba(255, 255, 255, 0.8)',  # Fancybox with transparency
    bordercolor='rgba(0, 0, 0, 0.5)',
    borderwidth=0
)

# Axes settings
template.layout.xaxis = dict(
    gridcolor='#cbcbcb',
    gridwidth=1,
    zeroline=False,
    ticklen=0,
    title=dict(font=dict(size=18)),  # x-large in Matplotlib corresponds to 18 in Plotly
    tickfont=dict(size=14, color='#808080')  # large in Matplotlib corresponds to 14 in Plotly
)

template.layout.yaxis = dict(
    gridcolor='#cbcbcb',
    gridwidth=1,
    zeroline=False,
    ticklen=0,
    title=dict(font=dict(size=18)),  # x-large in Matplotlib corresponds to 18 in Plotly
    tickfont=dict(size=14, color='#808080')  # large in Matplotlib corresponds to 14 in Plotly
)

# Color cycle
template.layout.colorway = ['#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c']

# Background color
template.layout.paper_bgcolor = '#f0f0f0'
template.layout.plot_bgcolor = '#f0f0f0'
template.layout.paper_bgcolor = template.layout.plot_bgcolor = '#ffffff'

# Font settings
template.layout.font = dict(
    size=14
)


# Figure margins
template.layout.margin = dict(
    l=80,  # 0.08 in Matplotlib corresponds to 80 in Plotly
    r=95,  # 0.95 in Matplotlib corresponds to 95 in Plotly
    b=70,  # 0.07 in Matplotlib corresponds to 70 in Plotly
    t=100,  # Additional top margin for title
    pad=5  # Padding between the margin and the plot
)

# Title settings
template.layout.title = dict(
    font=dict(size=20)  # x-large in Matplotlib corresponds to 20 in Plotly
)

template.layout.update(dict(width=800, height=600, autosize = False))

# Apply the template
pio.templates['fivethirtyeight'] = template
