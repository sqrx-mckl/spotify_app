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
from sklearn import config_context

from lib_spotify_app.model import (
    dbcv,
    validity_score,
    dbcv_validity_score,
    abs_dbcv_validity_score,
    make_processing,
    make_processing_parallel,
    make_search,
    make_optimization,
    make_default_search_param_spaces,
    make_default_optim_param_spaces,
    make_default_optim_param_spaces_parallel,
    analysis_plot_pipe,
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
df = pd.read_json('liked_songs.json')


# %%
n_samples = df.shape[0]/4
prior = 'uniform'

param_space = make_default_optim_param_spaces(n_samples, prior)
param_space_par = make_default_optim_param_spaces_parallel(n_samples, prior)


# %%
n_iter = 100
n_best = 5

# %% [markdown]
# # Runs different metric and scores

# %%
proc = make_processing(w_dist=False)

with config_context(display='diagram'):
    display(proc)

optimizer_serial = make_optimization(proc, param_space, n_iter=n_iter, scoring=dbcv_validity_score).fit(df)


# %%
analysis_plot_pipe(optimizer_serial, df, n_best=n_best)


# %%
proc_mah = make_processing(w_dist=True)

with config_context(display='diagram'):
    display(proc_mah)
    
optimizer_serial_mah = make_optimization(proc_mah, param_space, n_iter=n_iter, scoring=dbcv_validity_score).fit(df)


# %%
analysis_plot_pipe(optimizer_serial_mah, df, n_best=n_best)


# %%
proc_par = make_processing_parallel(w_dist=False, kwargs_umap={"metric":"euclidean"})

with config_context(display='diagram'):
    display(proc_par)

optimizer_par = make_optimization(proc_par, param_space_par, n_iter=n_iter, scoring=dbcv_validity_score).fit(df)


# %%
analysis_plot_pipe(optimizer_par, df, n_best=n_best)

# %% [markdown]
# Because of issues with HDBSCAN I use DBCV instead when combined with pre-computed.  
# `Mahalanobis` does not work directly, or at least not all properties such as `relative_validity` or `validity_index()`.

# %%
proc_mah = make_processing(w_dist=True)

with config_context(display='diagram'):
    display(proc_mah)

optimizer_serial_mah = make_optimization(proc_mah, param_space, n_iter=n_iter, scoring=dbcv).fit(df)


# %%
analysis_plot_pipe(optimizer_serial_mah, df, n_best=n_best)


# %%
proc_par_mah = make_processing_parallel(w_dist=True)#, kwargs_hdbscan={"metric":"precomputed"})

with config_context(display='diagram'):
    display(proc_par_mah)

optimizer_par_mah = make_optimization(proc_par_mah, param_space_par, n_iter=n_iter, scoring=dbcv_validity_score).fit(df)


# %%
analysis_plot_pipe(optimizer_par_mah, df, n_best=n_best)


# %%
proc_mah = make_processing(w_dist=True)

with config_context(display='diagram'):
    display(proc_mah)

def dbcv_validity_score2(pipe, X, y=None):
    return dbcv(pipe, X, y=None) + validity_score(pipe, X, y=None)
    
optimizer_serial_mah2 = make_optimization(proc_mah, param_space, n_iter=n_iter, scoring=dbcv_validity_score2).fit(df)


# %%
analysis_plot_pipe(optimizer_serial_mah2, df, n_best=n_best)

# %% [markdown]
# Up until now we used DBCV * Validity index, but it wasn't as satisfying.
# Below we try with absolute validty index.

# %%
proc_mah = make_processing(w_dist=True)

with config_context(display='diagram'):
    display(proc_mah)
    
optimizer_serial_mah3 = make_optimization(proc_mah, param_space, n_iter=n_iter, scoring=abs_dbcv_validity_score).fit(df)


# %%
analysis_plot_pipe(optimizer_serial_mah3, df, n_best=n_best)


# %%
proc_mah = make_processing(w_dist=True)

with config_context(display='diagram'):
    display(proc_mah)

def neg_dbcv_validity_score(pipe, X, y=None):
    return dbcv(pipe, X, y=None) * (-validity_score(pipe, X, y=None))

optimizer_serial_mah_neg = make_optimization(proc_mah, param_space, n_iter=n_iter, scoring=neg_dbcv_validity_score).fit(df)


# %%
analysis_plot_pipe(optimizer_serial_mah_neg, df, n_best=10, figsize=(12, 100))


# %%
def inv_dbcv_validity_score(pipe, X, y=None):
    return dbcv(pipe, X, y=None) * (1- abs(validity_score(pipe, X, y=None)))

optimizer_serial_mah4 = make_optimization(proc_mah, param_space, n_iter=n_iter, scoring=inv_dbcv_validity_score).fit(df)


# %%
analysis_plot_pipe(optimizer_serial_mah4, df, n_best=10, figsize=(12, 100))


# %%
from hdbscan.validity import validity_index

def find_clusterer(pipe):
    if "clusterer" in pipe.named_steps.keys():
        clusterer = pipe["clusterer"]
    else: #parallel
        named_steps = {t[0]:t[1] for t in pipe['transf'].transformer_list}
        clusterer = named_steps["clusterer"]
    return clusterer

def weighted_validity_score(pipe, X, y=None):
    clusterer = find_clusterer(pipe)
    X_map = X
    for name, estimator in pipe.steps[:-1]:
        if isinstance(estimator, str) or estimator is None:
            continue
        X_map = estimator.transform(X_map)
        if isinstance(X_map, pd.DataFrame):
            X_map = X_map.to_numpy()
        else:
            X_map = np.float64(X_map)
    vi = validity_index(
        X_map,
        clusterer.labels_,
        metric=clusterer.metric,
        per_cluster_scores=True
    )

    if vi[0] == 0:
        return 0

    vc = pd.Series(clusterer.labels_).value_counts(normalize=True, sort=False).values
    return np.mean(vi[1] * vc[1:])

optimizer_serial_mah5 = make_optimization(proc_mah, param_space, n_iter=n_iter, scoring=weighted_validity_score).fit(df)

# %%
analysis_plot_pipe(optimizer_serial_mah5, df, n_best=10, figsize=(12, 100))


# %%
optimizer_serial_mah6 = make_optimization(proc_mah, param_space, n_iter=n_iter, scoring=validity_score).fit(df)
analysis_plot_pipe(optimizer_serial_mah6, df, n_best=10, figsize=(12, 100))


