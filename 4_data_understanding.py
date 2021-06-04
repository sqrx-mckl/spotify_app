# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from IPython.display import display
from tqdm import tqdm

from sklearn.preprocessing import scale, minmax_scale
from scipy.spatial.distance import pdist, squareform
import umap
import umap.plot
import hdbscan

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16, 10)
sns.set_context('notebook')
sns.set_style('whitegrid')

# %% [markdown]
# # Data Collection

# %%
fp = Path(r'private/data.csv')


# %%
df = pd.read_csv(fp, sep='\t')


# %%
df.columns.to_list()


# %%
features = [
    'danceability',
    'energy',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    # 'popularity',
    # 'duration_min',
    # 'key'
]

X = df[features]

# %% [markdown]
# # Quick playlist creation

# %%
selection_df = df[df['danceability'] > df['danceability'].max() * 0.8]
selection_df = selection_df.sort_values(['valence'], ascending=True).head(50)
selection_df[['name', 'artists.name']]


# %%
import spotipy
import lib_spotify_app.api_adapter as api_adapter
import json

credential_path = Path(r'private/spotify_credential.json')

with open(str(credential_path)) as file:
    credential = json.load(file)

sp = api_adapter.setup_spotipy(
    credential_path,
    scope=['user-library-read','user-top-read', 'playlist-modify-public', 'playlist-modify-private'],
    cache_path=Path(r'private')
)


# %%
r_playlist = sp.user_playlist_create(credential['username'], 'test_danceability_high_valence_low')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for track_ids in chunks(selection_df['id'].to_list(), 50):
    sp.playlist_add_items(
        r_playlist['id'],
        track_ids,
    )

# %% [markdown]
# # Feature Analysis

# %%
X_proc = scale(X)


# %%
axs = sns.violinplot(x='variable', y='value', data=X.transform(minmax_scale).melt(), inner='box', color='white');
[ax.set_edgecolor('black') for ax in axs.collections];


# %%
X.plot(subplots=True, kind='box');


# %%
sns.clustermap(X.corr(method='spearman'), annot=True, vmin=-1, vmax=1, center=0);


# %%
from pandas.plotting import radviz

radviz(
    df[features+['artists.supergenre_1']],
    'artists.supergenre_1'
);

# %% [markdown]
# # Feature Importances

# %%
from sklearn.decomposition import PCA

pca = PCA(random_state=42).fit(X_proc)

plt.bar(x=np.arange(0, X.shape[1]), height=pca.explained_variance_ratio_);
plt.xlabel('#PC')
plt.ylabel('explained variance ratio')
plt.show();


# %%
pca_component_weighted = pd.DataFrame(pca.components_, columns=features).T
pca_component_weighted *= pca.explained_variance_ratio_
pca_component_weighted.plot(kind='barh', stacked=True, colormap='viridis');

# %% [markdown]
# # Dimensionality Reduction
# %% [markdown]
# ## Precompute distance matrix

# %%
dist_matrix = squareform(pdist(X, metric='mahalanobis'))
dist_matrix.shape

# %% [markdown]
# ## UMAP Projection
# 
# With default value, because some trials were done and shape does not change (n_neighbors distance the points between them)

# %%
mapper = umap.UMAP(
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    metric='precomputed',
    n_epochs=500,
    verbose=False,
).fit(dist_matrix)

# print(f'n_neighbors: {n_neighbors}\tmin_dist: {min_dist}')
umap.plot.points(mapper, labels=pd.factorize(df['artists.supergenre_1'])[0]);


# %%
umap.plot.diagnostic(mapper, diagnostic_type='vq');


# %%
# for k, feat_col in enumerate(tqdm(features)):
#     sns.scatterplot(
#         data=X,
#         x=proj_umap[:,0],
#         y=proj_umap[:,1],
#         hue=df[feat_col],
#         legend=None,
#         edgecolor=None,
#         ax=axes.flatten()[k]
#     ).set_title(feat_col)

for k, feature_value in enumerate(tqdm(X_proc.T)):
    ax = umap.plot.points(
        mapper,
        values=feature_value,
        theme='fire',
    );
    ax.set_title(features[k])


# %%
X_map = mapper.transform(dist_matrix)

# %% [markdown]
# # Clustering

# %%
clusterer_map = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    gen_min_span_tree=True,
    metric='euclidean',
    core_dist_n_jobs=4,
).fit(X_map)

# clusterer_raw = hdbscan.HDBSCAN(
#     min_cluster_size=50,
#     min_samples=None,
#     cluster_selection_epsilon=0.0,
#     metric='precomputed',
#     core_dist_n_jobs=4,
# ).fit(dist_matrix)


# %%
clusterer_map.single_linkage_tree_.plot(cmap='viridis', colorbar=True)


# %%
plt.scatter(
    x=X_map[:,0],
    y=X_map[:,1],
    s=clusterer_map.probabilities_,
    c=clusterer_map.labels_,
    cmap='Set2'
)
plt.colorbar();


# %%
np.unique(clusterer_map.labels_)


# %%
plt.boxplot(clusterer_map.cluster_persistence_);


# %%
clusterer_map.relative_validity_


# %%
np.unique([e.shape[0] for e in clusterer_map.exemplars_], return_counts=True)

# %% [markdown]
# # Create a Spotify Playlist

# %%
plt.scatter(
    x=X_map[clusterer_map.labels_==1,0],
    y=X_map[clusterer_map.labels_==1,1],
    c='r',
    cmap='Set2'
)
plt.scatter(
    x=X_map[clusterer_map.labels_!=1,0],
    y=X_map[clusterer_map.labels_!=1,1],
    c='grey',
    cmap='Set2'
)
plt.show();


# %%
df.loc[clusterer_map.labels_==1][['artists.name', 'name']]


# %%
import spotipy
import lib_spotify_app.api_adapter as api_adapter
import json

credential_path = Path(r'private/spotify_credential.json')

with open(str(credential_path))) as file:
    credential = json.load(file)

sp = api_adapter.setup_spotipy(
    credential_path,
    scope=['user-library-read','user-top-read', 'playlist-modify-public', 'playlist-modify-private'],
    cache_path=Path(r'private')
)


# %%
r_playlist = sp.user_playlist_create(credential['username'], 'test_instrumental')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

for track_ids in chunks(df.loc[clusterer_map.labels_==1]['id'].to_list(), 50):
    sp.playlist_add_items(
        r_playlist['id'],
        track_ids,
    )

# %% [markdown]
# # Optimization

# %%
X_map = squareform(pdist(X, metric='mahalanobis'))
X_dist = X_map


# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

dist_pipe = FunctionTransformer(
    lambda X: squareform(pdist(X, metric='mahalanobis'))
)

mapper = umap.UMAP(
    random_state=42,
    n_neighbors=15,
    min_dist=0.1,
    metric='precomputed',
    n_epochs=500,
    verbose=False,
)

clusterer_map = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=None,
    gen_min_span_tree=True,
    metric='euclidean',
    core_dist_n_jobs=1,
)

proc = Pipeline([
    ('dist', dist_pipe),
    ('mapper', mapper),
    ('clusterer', clusterer_map)
])


# %%
best_proc = bscv.best_estimator_
display(bscv.best_params_)


# %%
best_proc = bscv.best_estimator_
display(bscv.best_params_)


# %%
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import ShuffleSplit

search_spaces = {
    'mapper__min_dist': Real(10e-6, 0.99, prior='log-uniform'),
    'mapper__n_neighbors': Integer(5, 1000, prior='log-uniform'),
    'clusterer__min_cluster_size': Integer(5, 1000, prior='log-uniform'),
    'clusterer__min_samples': Integer(5, 1000, prior='log-uniform'),
}

def dbcv(pipe:Pipeline, X):
    return pipe['clusterer'].relative_validity_

base_estimator='GBRT',
bscv = BayesSearchCV(
    proc,
    search_spaces,
    scoring=dbcv,
    n_iter=1000,
    cv=ShuffleSplit(n_splits=1, random_state=42),
    verbose=2,
    return_train_score=True,
    refit=True,
    random_state=42,
    optimizer_kwargs=dict(initial_point_generator='lhs'),
    n_points=4,
    n_jobs=4,
).fit(X)


# %%
r = pd.DataFrame(bscv.cv_results_)    .sort_values('mean_train_score', ascending=False)
r = r.loc[:, r.columns.str.contains('param|mean_train_score')]     .drop_duplicates(subset=r.columns[r.columns.str.contains('param_')])

display(r.shape)

r.head(10).style.bar(subset='mean_train_score')


# %%
for k, params in enumerate(tqdm(r['params'].iloc[:10])):

    estim = proc.set_params(**params).fit(X)
    X_map = X
    
    for name, estimator in estim.steps[:-1]:
        X_map = estimator.transform(X_map)

    _, ax = plt.subplots()
    plt.scatter(
        x=X_map[:,0],
        y=X_map[:,1],
        s=estim['clusterer'].probabilities_*10,
        c=estim['clusterer'].labels_,
        cmap='Set2'
    )
    plt.colorbar();
    ax.set_title(f'param nb {k}: UMAP projection clustered')

plt.show();


# %%
# _, axs = plt.subplots(1, 2)
# axs[0].hist(best_proc['clusterer'].labels_, density=True)
# axs[0].set_title('clusters density')

# sns.histplot(
#     x=best_proc['clusterer'].probabilities_,
#     hue=best_proc['clusterer'].labels_,
#     ax=axs[1],
# )
# axs[1].set_title('samples cluster probability')

# _ = plt.show()


# %%
np.geomspace(5, 1000, 3, dtype=int)


# %%
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.pipeline import Pipeline

X_dist = squareform(pdist(X, metric='mahalanobis'))

def dbcv(pipe:Pipeline, X=None, y=None):
    return pipe['clusterer'].relative_validity_

def dbcv_normalized(pipe:Pipeline, X=None, y=None):
    return pipe['clusterer'].relative_validity_ / np.log1p(pipe['clusterer'].labels_.max())

def n_clusters(pipe:Pipeline, X=None, y=None):
    return pipe['clusterer'].labels_.max()

n_search = 1

param_spaces = {
    'mapper__min_dist': np.geomspace(10e-6, 0.99, n_search),
    'mapper__n_neighbors': [int(x) for x in np.geomspace(5, 1000, n_search)],
    'clusterer__min_cluster_size': [int(x) for x in np.geomspace(5, 1000, n_search)],
    'clusterer__min_samples': [int(x) for x in np.geomspace(5, 1000, n_search)],
}

r_grid = {
    'params': [],
    'dbcv': [],
    'dbcv_norm': [],
    'n_clusters': [],
}

pipe = Pipeline([
    ('mapper', mapper),
    ('clusterer', clusterer_map)
])

for params in ParameterGrid(param_spaces):
    pipe = pipe.set_params(**params).fit(X_dist)
    r_grid['params'].append(params)
    r_grid['dbcv'].append(dbcv(pipe))
    r_grid['dbcv_norm'].append(dbcv_normalized(pipe))
    r_grid['n_clusters'].append(n_clusters(pipe))

r_grid = pd.DataFrame.from_dict(r_grid)
r_grid


# %%
n_search = 3

param_spaces = {
    'mapper__min_dist': np.geomspace(10e-6, 0.99, n_search),
    'mapper__n_neighbors': [int(x) for x in np.geomspace(5, 1000, n_search)],
    'clusterer__min_cluster_size': [int(x) for x in np.geomspace(5, 1000, n_search)],
    'clusterer__min_samples': [int(x) for x in np.geomspace(5, 1000, n_search)],
}

gscv = GridSearchCV(
    Pipeline([('mapper', mapper), ('clusterer', clusterer_map)]),
    param_spaces,
    scoring={'DBCV':dbcv, 'DBCV_norm':dbcv_normalized, 'n_clusters':n_clusters},
    cv = [(slice(None), slice(None))],
    verbose=2,
    return_train_score=False,
    refit=False,
    n_jobs=4,
).fit(X_dist)

pd.DataFrame(gscv.cv_results_)


# %%
r_lhs = pd.DataFrame(gscv.cv_results_)    .sort_values('mean_train_DBCV', ascending=False)
r_lhs = r_lhs.loc[:, r_lhs.columns.str.contains('param_|mean_')]     .drop_duplicates(subset=r_lhs.columns[r_lhs.columns.str.contains('param_')])

display(r_lhs.shape)

r_lhs.head(20).style.bar(subset=['mean_train_DBCV', 'mean_train_DBCV_norm', 'mean_train_n_clusters'])


# %%
import plotly.express as px

for score in ['mean_train_DBCV', 'mean_train_DBCV_norm', 'mean_train_n_clusters']:
    fig = px.parallel_coordinates(
        r_lhs,
        color=score,
        dimensions=r_lhs.columns[r_lhs.columns.str.contains('param_')].to_list(),
        color_continuous_scale=px.colors.diverging.Tealrose,
    )
    fig.show()


# %%
best_proc_ = proc()
display(gscv.best_params_)

X_map = X
for name, estimator in best_proc_norm.steps[:-1]:
    X_map = estimator.transform(X_map)


# %%



