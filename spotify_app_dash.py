#%% Imports

import dash_html_components as html
import dash_core_components as dcc
import dash
import plotly.graph_objects as go
import umap
import yaml
import hdbscan
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Dict, List, Union
import pandas as pd
import lib_spotify_app as lib
import numpy as np
from pathlib import Path
from pprint import pprint
from IPython.display import display
import seaborn as sns

# Settings
plt.ioff()
sns.set_style('whitegrid')
sns.set_context('notebook')
pd.set_option('max_columns', None)

#%% Load full dataset

df:pd.DataFrame = pd.read_parquet(
    Path('private', 'df_analysis_dbscan.parquet'))

#%% Prepare for plot

# Retrieve the most "popular" supergenre
supergenres_best = lib.EnrichArtistGenre.get_df_genre(
    df.index,
    df['track.artists.0.supergenres']
)[0].sum()

def keep_best_genre(x)->str:
    supergenres = [word for word in x if word != "outliers"]
    if len(supergenres) == 0:
        return "outliers"
    else:
        return supergenres_best[supergenres].idxmax()

df['supergenres_str'] = df['track.artists.0.supergenres'].apply(
    keep_best_genre
)

# Easier one
df['genres_str'] = df['track.artists.0.genres'].apply(
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

col_analysis = [
    'track.id.danceability',
    'track.id.energy',
    'track.id.key',
    'track.id.loudness',
    'track.id.mode',
    'track.id.speechiness',
    'track.id.acousticness',
    'track.id.instrumentalness',
    'track.id.liveness',
    'track.id.valence',
    'track.id.tempo',
    'track.id.time_signature',
]

plt.figure()
sns.pairplot(df,
            kind='scatter',
            diag_kind='hist',
            corner=True,
            vars=col_analysis)
plt.savefig(Path('private', f'pairplot.png'), bbox_inches = 'tight')
plt.show()

#%% Cluster

df[['tsne_x', 'tsne_y']] = pd.DataFrame(TSNE(perplexity=10, learning_rate=200)\
                                        .fit_transform(df[col_analysis]))

df[['umap_x', 'umap_y']] = pd.DataFrame(umap\
                                .UMAP(n_neighbors=15, metric='mahalanobis')\
                                .fit_transform(df[col_analysis]))

for algo in ['tsne', 'umap']:
    df[f'clusters_{algo}'] = hdbscan.HDBSCAN(min_cluster_size=5)\
                                    .fit_predict(df[[f'{algo}_x', f'{algo}_y']])
    
    df[f'clusters_{algo}'] = df[f'clusters_{algo}'].apply(
        lambda x: f'c{x}'
    )

#%% Quick plots to check
for algo in ['tsne', 'umap']:
    plt.figure()
    df[f'clusters_{algo}'].value_counts()\
                        .plot(kind='barh')\
                        .set_title(algo)
    plt.show()

for algo in ['tsne', 'umap']:
    plt.figure()
    sns.scatterplot(f'{algo}_x', f'{algo}_y',
                    data=df,
                    size='track.popularity',
                    hue=f'supergenres_str')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

#%% Pairplot, quite slow
if False:
    for algo in ['tsne', 'umap']:
        plt.figure()
        sns.pairplot(df,
                    kind='scatter',
                    diag_kind='hist',
                    corner=True,
                    vars=col_analysis + [f'{algo}_x', f'{algo}_y'],
                    hue=f'clusters_{algo}')
        plt.savefig(Path('private', f'pairplot_{algo}.png'),
                    bbox_inches = 'tight')
        plt.show()

#%% Last FM

last_api = lib.LastAPI(Path('private', 'lastfm_credential.json'))
top_tracks_lastfm = pd.DataFrame(
    last_api.user.get_top_tracks(period='overall', limit=500)
)
display(top_tracks_lastfm)

# #%% Dash/Plotly

# mdl_plt = 'tsne'

# fig = lib.plotly_categorical_scatter(
#     df,
#     x=f'{mdl_plt}_x',
#     y=f'{mdl_plt}_y',
#     hue=f'clusters_{mdl_plt}',
#     size='size',
#     text='text',
#     link='track.external_urls.spotify'
# )
# fig.show()

# # %%

# app = dash.Dash()
# app.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])

# app.run_server(debug=True, use_reloader=False)
