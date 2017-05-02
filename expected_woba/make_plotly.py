#!/usr/bin/python
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot
import pandas as pd

#%%
df = pd.read_csv(r'eWOBA.csv')

plot_data = []

teams = np.sort(df.Team.unique())
for team in teams:
    xdata = df.loc[df.Team==team]['wOBA']
    ydata = df.loc[df.Team==team]['proj_wOBA']
    text = df.loc[df.Team==team]['Name']
    
    t0 = go.Scatter(
      x = xdata,
      y = ydata,
      mode='markers',
      name=team,
      text=text,
      hoverinfo='text'
    )
    plot_data.append(t0)
    
t1 = go.Scatter(
  x = [0.1,0.55],
  y = [0.1,0.55],
  mode = 'lines',
  name = 'y=x'        
)
plot_data.append(t1)
    
layout = go.Layout(
    title='Projected wOBA vs. Actual',
    xaxis=dict(
        title='wOBA'
    ),
    yaxis=dict(
        title='Projected wOBA'
    ),
    hovermode="closest"
)

fig = go.Figure(data=plot_data, layout=layout)
#plotly_plot(fig,filename='scatter.html')

#%%
# Better way that calls plotly library from CDN
div = plotly_plot(fig, include_plotlyjs=False, output_type='div')
html = '''
<html>
<head>
 <meta charset="utf-8" />
 <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
''' + div + '</body></html>'
with open('scatter.html','wb') as fid:
    fid.write(html)
