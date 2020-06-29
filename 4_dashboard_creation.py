
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
