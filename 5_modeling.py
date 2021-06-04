# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from IPython.display import display
from tqdm import tqdm

from lib_spotify_app.model import (
    make_processing,
    make_search,
    make_optimization
)

from lib_spotify_app.api_adapter import (
    make_spotify_playlist,
    setup_spotipy,
    get_credential,
    query_liked_songs,
    enrich_audiofeature,
    normalize_request
)

from lib_spotify_app.enrich_artist_genre import add_genres

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16, 10)
sns.set_context('notebook')
sns.set_style('whitegrid')


# %%
credential_path = Path(r'private/spotify_credential.json')
credential = get_credential(credential_path)


# %%
sp = setup_spotipy(
    credential_path,
    scope=['user-library-read','user-top-read', 'playlist-modify-public', 'playlist-modify-private'],
    cache_path=Path(r'private')
)


# %%
request = sp.playlist_tracks('2HKu6hhXqMU6qbYhEWjPmt')


# %%
df = normalize_request(request)


# %%
df = enrich_audiofeature(df, sp, col="track.id")
df, genre = add_genres(df, sp)


# %%
from sklearn.preprocessing import MultiLabelBinarizer

MultiLabelBinarizer().fit_transform(df['artists.genres'])


# %%
df.columns.tolist()


# %%
from sklearn import config_context
proc = make_processing(kwargs_umap={"n_epochs":100})

with config_context(display='diagram'):
    display(proc)


# %%
proc.fit(df)


# %%
search = make_search(proc=proc, param_spaces=5)


# %%
search.fit(df)


# %%
import plotly.express as px

X_parallel_coor = [
    "param_clusterer__min_cluster_size",
    "param_clusterer__min_samples",
    "param_mapper__min_dist",
    "param_mapper__n_neighbors"
]

y_parallel_coor = [
    'mean_test_n_clusters',
    "mean_test_DBCV",
    "mean_test_DBCV_norm",
    "mean_test_DBCV_mult",
]


# %%
r = pd.DataFrame(search.cv_results_).drop('params', axis=1)
r[X_parallel_coor+y_parallel_coor]    .sort_values('mean_test_n_clusters', ascending=False)    .head(10)    .style.background_gradient()


# %%
for X in X_parallel_coor:
    plt.figure()
    pd.plotting.parallel_coordinates(r, X, cols=y_parallel_coor)


# %%
for y in y_parallel_coor:
    plt.figure()
    pd.plotting.parallel_coordinates(r, y, cols=X_parallel_coor)


# %%
for X in X_parallel_coor:
    fig = px.parallel_coordinates(
        r,
        color=X,
        dimensions=y_parallel_coor,
    )
    fig.show()


# %%
optimizer = make_optimization(proc, n_iter=100)


# %%
optimizer.fit(df)


# %%
r_opt = pd.DataFrame(optimizer.cv_results_).drop('params', axis=1)
r_opt[X_parallel_coor+y_parallel_coor]    .assign(dbcv_mult=lambda x: x['mean_test_DBCV'] * x['mean_test_n_clusters'])    .sort_values('mean_test_n_clusters', ascending=False)    .head(10)    .style.background_gradient()


# %%



