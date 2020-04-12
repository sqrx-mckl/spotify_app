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
sns.set_style('whitegrid')
sns.set_context('notebook')

pd.set_option('max_columns', None)

#%% Load full dataset

if False:
    df_likedsong_enrich = pd.read_csv(Path('private', 'likedsongs_enrich.csv'),
                                    index_col=0)
    df: pd.DataFrame = df_likedsong_enrich[[
        'added_at',
        'track.album.id',
        'track.album.images.0.height',
        'track.album.images.0.url',
        'track.album.images.0.width',
        'track.album.name',
        'track.album.release_date',
        'track.album.release_date_precision',
        'track.album.total_tracks',
        'track.artists.0.id',
        'track.artists.0.name',
        'track.duration_ms',
        'track.id',
        'track.external_urls.spotify',
        'track.name',
        'track.popularity',
        'track.preview_url',
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
        'track.artists.0.genres',
        'track.artists.0.supergenres'
    ]]
else:
    df: pd.DataFrame = pd.read_csv(Path('private', 'df_analysis.csv'),
                                index_col=0)

    col_to_convert = ['track.artists.0.genres', 'track.artists.0.supergenres']
    df[col_to_convert] = df[col_to_convert].applymap(yaml.safe_load)


#%% Genre again

genres = lib.EnrichArtistGenre(df['track.artists.0.id'],
                                genre=df['track.artists.0.genres'])

mask = lib.mask_outlier_iqr(genres.df_genre.sum())

plt.figure
genres.df_genre.loc[mask.low].plot(kind='box')
plt.figure
genres.df_genre.loc[mask.low].plot(kind='hist')

genres.df_genre = genres.df_genre.iloc[:, mask]

genres.cluster_genre_fit(algorithm='hdbscan')

plt.figure()
g = sns.countplot(genres.cluster_dbscan)
g.set_title('DBscan clusters histogram')
display(genres._get_group_list(genres.cluster_dbscan))

genres.cluster_genre_transform(t=40, criterion='maxclust', verbose=True)
genres.plot_chord_supergenre()

#%%

plt.figure()
genres.df_genre.sum().plot(kind='box')

plt.figure()
genres.df_genre.sum().plot(kind='hist')

plt.figure()
genres.df_genre.sum().sort_values().plot(kind='barh')

#%%
supergenre_corr = genres.df_supergenre.corr(method=lambda x, y: max(x*y))
sns.heatmap(supergenre_corr)

# I want to know which supergenre is never alone
genres.df_supergenre.corr(method=lambda x, y: sum(x*y))

supergenres_encounter = genres.df_supergenre.agg(sum, axis=1)
genres.df_supergenre.agg(
    lambda x: np.mean(supergenres_encounter[x>0])
)

#%% Cluster

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

df[['tsne_x', 'tsne_y']] = pd.DataFrame(TSNE(perplexity=15, learning_rate=400)\
                                        .fit_transform(df[col_analysis]))

df[['umap_x', 'umap_y']] = pd.DataFrame(umap\
                                .UMAP(n_neighbors=15, metric='mahalanobis')\
                                .fit_transform(df[col_analysis]))

for algo in ['tsne', 'umap']:
    df[f'clusters_{algo}'] = hdbscan.HDBSCAN(min_cluster_size=20)\
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

for algo in ['tsne', 'umap']:
    plt.figure()
    sns.scatterplot(f'{algo}_x', f'{algo}_y',
                    data=df,
                    hue=f'clusters_{algo}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

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

#%% Prepare for dash
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


#%% Dash/Plotly

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
