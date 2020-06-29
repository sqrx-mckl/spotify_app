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
import lib_spotify_app.facade_enrich_artist_genre as lib
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

#%% Cluster genre again
genres.cluster_genre_fit(algorithm='hdbscan')

plt.figure()
sns.countplot(genres.cluster_dbscan).set_title('DBscan clusters histogram')

display(genres._get_group_list(genres.cluster_dbscan))

#%%

def save_df(df, genres, name):
    df = df.drop(['track.artists.0.genres',
            'track.artists.0.supergenres'], axis=1)
    df = genres.enrich_df(df)

    df.to_parquet(Path('private', f'df_analysis_{name}.parquet'))


#%% HDBSCAN only
genres.cluster_genre_transform_dbscan(lof=True, verbose=True)
genres.plot_chord_supergenre()

plt.figure()
genres.supergenre_occurences.plot(kind='box')

save_df(df, genres, 'dbscan')

#%% manuel

for n_cluster in [5, 10, 20]:
    genres.cluster_genre_transform(t=n_cluster, verbose=True)
    genres.plot_chord_supergenre()

    save_df(df, genres, f'manual{n_cluster}')

# %%
