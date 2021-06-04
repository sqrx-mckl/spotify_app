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
# # Optimization

# %%
X_map = squareform(pdist(X, metric='mahalanobis'))

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import umap
import hdbscan

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
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics.cluster import silhouette_score

def dbcv(estimator, X, y=None):
    return estimator.relative_validity_

def dbcv_norm(estimator, X, y=None):
    return estimator['clusterer'].relative_validity_ / np.log1p(estimator['clusterer'].labels_.max())

def n_clusters(estimator, X, y=None):
    return estimator['clusterer'].labels_.max()

n_search = 1

param_spaces = {
    'mapper__min_dist': np.geomspace(10e-6, 0.99, n_search),
    'mapper__n_neighbors': np.geomspace(5, 1000, n_search).astype(int),
    'clusterer__min_cluster_size': np.geomspace(5, 1000, n_search).astype(int),
    'clusterer__min_samples': np.geomspace(5, 1000, n_search).astype(int),
}

gscv = GridSearchCV(
    clusterer_map,
    # Pipeline([('mapper', mapper), ('clusterer', clusterer_map)]),
    {
        'min_cluster_size': np.geomspace(5, 1000, n_search).astype(int),
        'min_samples': np.geomspace(5, 1000, n_search).astype(int)
    },
    scoring=dbcv,#{'DBCV':dbcv, 'DBCV_norm':dbcv_norm, 'n_clusters':n_clusters},
    cv=[(slice(None), slice(None))],
    verbose=2,
    refit=False,
    n_jobs=1,
).fit(X)

display(pd.DataFrame(gscv.cv_results_))

# %%
gscv.estimator['clusterer'].relative_validity_

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
