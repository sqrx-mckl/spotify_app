# %%
import dash_html_components as html
import dash_core_components as dcc
import dash
import plotly.graph_objects as go
from sklearn.neighbors import DistanceMetric
import umap
import yaml
import hdbscan
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
from scipy.spatial.distance import yule
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Dict, List, Union
import pandas as pd
import lib_spotify_app as lib
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


pd.set_option('max_columns', None)
pd.reset_option('max_rows')


# %%
sp_adapter = lib.adapter_spotipy_api(
    credential_fp=Path(r'private/spotify_credential.json'),
    scope=' '.join(['user-library-read', 'user-top-read', ]),
    cache_path=Path(r'private')
)

sp_adapter.open_session()

# %%

df_likedsong: pd.DataFrame = lib.normalize_request(
    sp_adapter.query_liked_songs()
).pipe(lib.enrich_audiofeature, col='track.id')

df_likedsong.to_csv(Path('private', 'likedsongs.csv'))

# %%

df_likedsong: pd.DataFrame = pd.read_csv(Path('private', 'likedsongs.csv'))

df_likedsong['added_at'] = pd.to_datetime(df_likedsong['added_at'])

# plot the amount of time I added a song
df_likedsong['added_at']\
    .groupby(by=df_likedsong['added_at'].dt.date)\
    .count()\
    .plot()\
    .set_title('How many songs I liked per day')

df_likedsong = df_likedsong.drop('Unnamed: 0', axis=1)

# %%

genre_likedsong = lib.facade_enrich_artist_genre(
    df_likedsong['track.artists.0.id'],
    sp=sp_adapter.sp
)

# %%

genre_likedsong.df_genre = genre_likedsong._df_genre_raw
genre_likedsong.clean_useless_genre()
pprint(genre_likedsong._genre_cleaned.to_list())
pprint(genre_likedsong.genre.to_list())

# %%

genre_likedsong.cluster_genre_fit(method='average')

# genre_likedsong.plot_clustermap()
genre_likedsong.plot_dendrogram()

# %%

genre_likedsong.cluster_genre_transform(t=100, criterion='maxclust')
genre_likedsong._analyse_supergenre()
print(f'there is {len(genre_likedsong.supergenre)} supergenres')
# genre_likedsong.plot_heatmap_supergenre()

# %%

genre_likedsong.test_supergenre_distance()
genre_likedsong.test_supergenre_maxclust()

# %%

genre_likedsong.cluster_genre_transform_auto(verbose=True)

# %%


mdl_hdbscan = hdbscan.HDBSCAN(
    min_cluster_size=2,
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.5,
    min_samples=5,
    metric='yule'
)

cluster_hdbscan = mdl_hdbscan.fit_predict(
    genre_likedsong.df_supergenre.transpose()
)

display(genre_likedsong._get_group_list(cluster=cluster_hdbscan,
                                        df_genre=genre_likedsong.df_supergenre))

# %%
df_likedsong_enrich = genre_likedsong.enrich_df(df_likedsong)
display(len(df_likedsong_enrich))
display(df_likedsong_enrich.sample(5))

# %%

df_likedsong_enrich.to_csv(Path('private', 'likedsongs_enrich.csv'))

# %%

df_likedsong_enrich = pd.read_csv(Path('private', 'likedsongs_enrich.csv'),
                                  index_col=0)

# %%

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

# %%
df.to_csv(Path('private', 'df_analysis.csv'))

# %%
df: pd.DataFrame = pd.read_csv(Path('private', 'df_analysis.csv'),
                               index_col=0)

col_to_convert = ['track.artists.0.genres', 'track.artists.0.supergenres']
df[col_to_convert] = df[col_to_convert].applymap(yaml.safe_load)

# %%

df_num = df.select_dtypes('number')

fig, ax = plt.subplots(df_num.shape[1], 1,
                       figsize=(20, 20),
                       gridspec_kw={'hspace': 1})
for k, col in enumerate(df_num):
    df_num.boxplot(column=col, vert=False, ax=ax[k])

df_num.hist(figsize=(20, 14))

# %%

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

# def hdbscan_fit_transform(min_cluster_size, df):
#     mdl = hdbscan.HDBSCAN(
#         min_cluster_size=20,
#         metric='mahalanobis',
#         V=df.cov(),
#         gen_min_span_tree=True)

#     df_output = mdl.fit_predict(df)
#     # mdl.minimum_spanning_tree_.plot()
#     # mdl.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
#     return df_output

# mdl_tsne = TSNE(metric=DistanceMetric.get_metric('mahalanobis',
#                                                 V=df_analysis.cov()))\
#                 .fit_transform(df[col_analysis])


df[['tsne_x', 'tsne_y']] = pd.DataFrame(TSNE(perplexity=30, learning_rate=600)\
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

#%% check the outcome of the clustering algorithm
for algo in ['tsne', 'umap']:
    plt.figure()
    df[f'clusters_{algo}'].value_counts()\
                          .plot(kind='barh')\
                          .set_title(algo)

    # plt.figure()
    # df[f'clusters_{algo}'].hist().set_title(algo)

#%% Convert clusters to categorical


# %%
# From now on I will plot the data

# %% [markdown]
# Seaborn plot with first the dataa 

#%%
for algo in ['tsne', 'umap']:
    plt.figure()
    sns.scatterplot(f'{algo}_x', f'{algo}_y',
                    data=df,
                    hue=f'clusters_{algo}')

#%%
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
    plt.close()

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

# %%
