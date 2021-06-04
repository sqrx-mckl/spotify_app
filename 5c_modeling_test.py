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
import plotly.express as px

from lib_spotify_app.model import (
    dbcv,
    validity_score,
    dbcv_validity_score,
    abs_dbcv_validity_score,
    make_processing,
    make_search,
    make_optimization,
    make_default_search_param_spaces,
    make_default_optim_param_spaces,
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
credential_path = Path(r'private/sqrx_credential.json')
credential = get_credential(credential_path)


# %%
sp = setup_spotipy(
    credential_path,
    scope=['user-library-read','user-top-read', 'playlist-modify-public', 'playlist-modify-private'],
    cache_path=Path(r'private')
)


# %%
n_iter = 100
n_best = 5


# %%
def get_data(playlist_uri) -> pd.DataFrame:
    request = sp.playlist_tracks(playlist_uri)
    df = normalize_request(request)
    df = enrich_audiofeature(df, sp, col="track.id")
    df, genre = add_genres(df, sp)
    return df, genre


# %%
df1, _ = get_data('37i9dQZF1DXca8AyWK6Y7g')
df2, _ = get_data('37i9dQZF1DWTBN71pVn2Ej')
df3, _ = get_data('37i9dQZF1DX9Mqxt6NLTDY')
df = pd.concat([df1, df2, df3])

# df.to_json('playlist_young_and_free.json')
# df


# %%

from lib_spotify_app.model import find_clusterer

def dbcv(pipe, X=None, y=None):
    clusterer = find_clusterer(pipe)
    score = clusterer.relative_validity_
    print(score)
    if score == 0:
        return np.random.random_integers(10) * 1e-6
    return score

proc_mah = make_processing(w_dist=True)
display(proc_mah)

def get_optimal(df, n_iter=n_iter, scoring=dbcv):
    param_space = make_default_optim_param_spaces(df.shape[0]//4, 'uniform')
    print(param_space)
    return make_optimization(
        proc_mah,
        param_space,
        n_iter=n_iter,
        scoring=dbcv,
        n_jobs=1
    ).fit(df)

optim = get_optimal(df, scoring=dbcv)
analysis_plot_pipe(optim, df, n_best=n_best)


# %%



