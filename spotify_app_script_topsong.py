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
