
# %% PlotLy

# %%

df['genres_str'] = df['track.artists.0.genres'].apply(
    lambda x: ' '.join(x)
)
df['supergenres_str'] = df['track.artists.0.supergenres'].apply(
    lambda x: ' '.join(x)
)

df['text'] = df.apply(lambda x:
                      f'Track: {x["track.name"]}<br>' +
                      f'Artist: {x["track.artists.0.name"]}<br>' +
                      f'Album: {x["track.album.name"]}<br>' +
                      f'Release: {x["track.album.release_date"]}<br>' +
                      # f'Genre: {x["genres_str"]}<br>'+
                      f'Super Genre: {x["supergenres_str"]}<br>' +
                      f'Preview Song: <a href="{x["track.preview_url"]}">Play</a><br>' +
                      f'Full Song: <a href="{x["track.external_urls.spotify"]}">Play</a><br>', axis=1)

df['size'] = df['track.popularity'].apply(lambda x: np.log10(x+1))

#%%

mdl_plt = 'tsne'

fig = lib.plotly_categorical_scatter(
    df,
    x=f'{mdl_plt}_x',
    y=f'{mdl_plt}_y',
    hue=f'clusters_{mdl_plt}',
    size='size',
    text='text',
    link='track.external_urls.spotify'
)
fig.show()

# %%

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=False)

import webbrowser
import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

def click_event(trace, points, state):
    print('test')
    display(points.customdata)
    [webbrowser.open(point.customdata) for point in points]


fig = go.FigureWidget()

for group_name, df_group in df_top5.groupby('clusters'):
    fig.add_trace(go.Scattergl(
        x=df_group['proj_x'],
        y=df_group['proj_y'],
        name=group_name,
        text=df_group['text'],
        marker_size=df_group['size'],
        customdata=df_group['preview_url'],
        ids=df_group.index.to_list(),
        marker=dict(
            color=group_name,
            colorscale='Safe',
            line_width=0
        )
    ))
    
# Tune marker appearance and layout
fig.update_layout(autosize=False, width=640, height=480)

# upadte the size of the points based on the overall size of all traces
fig.update_traces(
    mode='markers',
    hoverinfo='text',
    marker=dict(sizeref=max(df_top5['size'])/10.)
)
fig.for_each_trace(
        lambda trace: trace.on_click(click_event, append=True)
)