#%%
import numpy as np
from pathlib import Path
import spotipy
import json
from pprint import pprint
from IPython.display import display

import dill

import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('notebook')

import lib_spotify_app as lib

import pandas as pd

pd.set_option('max_columns', None)
pd.reset_option('max_rows')

from typing import Dict, List, Union

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import yule

from scipy.cluster.hierarchy import fcluster, dendrogram, linkage

#%%
sp_adapter = lib.adapter_spotipy_api(
    credential_fp=Path(r'private/spotify_credential.json'),
    scope=' '.join(['user-library-read','user-top-read',]),
    cache_path=Path(r'private')
)

sp_adapter.open_session()

#%% [markdown]
# As an example I can retrieve my top artists list and do some plots of it.

#%% Query / Request
top_artists = lib.normalize_request(
    sp_adapter.sp.current_user_top_artists(limit=20)
)

#%% Plot

display(top_artists[['name', 'popularity', 'followers.total']])

top_artists_header = top_artists.name.to_frame().apply(
    lambda x: f'{x.name}: {x.values[0]}', axis=1
)

top_artists[['popularity', 'followers.total']]\
    .assign(top=top_artists_header)\
    .plot(x='top',
          kind='bar',
          secondary_y='followers.total')

top_artists[['popularity', 'followers.total']]\
    .assign(top=top_artists_header)\
    .plot(x='top',
          kind='barh',
          subplots=True,
          layout=(1,2),
          sharey=True,
          sharex=False)

# %%

df_toptracks = lib.normalize_request(
    sp_adapter.sp.current_user_top_tracks(limit=50)
).pipe(lib.enrich_audiofeature, col='id')

# %%

genre_toptracks = lib.facade_enrich_artist_genre(
    artists=df_toptracks['artists.0.id'],
    sp=sp_adapter.sp
)

#%%
genre_toptracks.cluster_genre(method='weighted',
                              fit=True,
                              verbose=True)

#%%

genre_toptracks.test_supergenre_distance()
genre_toptracks.test_supergenre_maxclust()

#%%

df_likedsong:pd.DataFrame = lib.normalize_request(
    sp_adapter.query_liked_songs()
).pipe(lib.enrich_audiofeature, col='track.id')

df_likedsong.to_csv(Path('private', 'likedsongs.csv'))

df_likedsong['added_at'] = pd.to_datetime(df_likedsong['added_at'])

# plot the amount of time I added a song
df_likedsong['added_at']\
            .groupby(by=df_likedsong['added_at'].dt.date)\
            .count()\
            .plot()\
            .set_title('How many songs I liked per day')

#%%

genre_likedsong = lib.facade_enrich_artist_genre(
    df_likedsong['track.artists.0.id'],
    sp=sp_adapter.sp
)

#%%

genre_likedsong.df_genre = genre_likedsong._df_genre_raw
genre_likedsong.clean_useless_genre()
pprint(genre_likedsong._genre_cleaned.to_list())
pprint(genre_likedsong.genre.to_list())

#%%

genre_likedsong.cluster_genre_fit(method='average')

# genre_likedsong.plot_clustermap()
genre_likedsong.plot_dendrogram()

#%%

genre_likedsong.cluster_genre_transform(t=100, criterion='maxclust')
genre_likedsong._analyse_supergenre()
print(f'there is {len(genre_likedsong.supergenre)} supergenres')
# genre_likedsong.plot_heatmap_supergenre()

#%%

genre_likedsong.test_supergenre_distance()
genre_likedsong.test_supergenre_maxclust()

# %%

# dill.dump_session(Path('private', 'session_dump_before_plot'))

# %%

genre_likedsong.cluster_genre_transform_auto(verbose=True)

# %%
df_likedsong_enrich = genre_likedsong.enrich_df(df_likedsong)
display(df_likedsong_enrich.sample(20))

# %%

df_likedsong_enrich.to_csv(Path('private', 'likedsongs_enrich.csv'))

# %%

df_likedsong_enrich = pd.read_csv(Path('private', 'likedsongs_enrich.csv'))

#%%

df = df_likedsong_enrich[[
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

df.to_csv(Path('private', 'df_analysis.csv'))

df:pd.DataFrame = pd.read_csv(Path('private', 'df_analysis.csv'))

# %%

df_num = df.select_dtypes('number')

fig, ax = plt.subplots(df_num.shape[1], 1,
                        figsize=(20, 20),
                        gridspec_kw={'hspace': 1})
for k, col in enumerate(df_num):
    df_num.boxplot(column=col, vert=False, ax=ax[k])

df_num.hist(figsize=(20,14))

#%%
# From now on I will plot the data
import plotly.graph_objects as go

fig = go.Figure()

# %%
