import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from hdbscan.validity import validity_index

import matplotlib.pyplot as plt

from lib_spotify_app.api_adapter import (
    enrich_audiofeature,
)

from lib_spotify_app.enrich_artist_genre import add_genres

__all__ = [
    "FeatureSelector",
    "make_processing",
    "make_processing_parallel",
    "make_default_search_param_spaces",
    "make_default_optim_param_spaces"
    "make_default_optim_param_spaces_parallel",
    "make_search",
    "make_optimization",
    "dbcv",
    "dbcv_normalized",
    "dbcv_mult_n_clusters",
    "dbcv_validity_score",
    "validity_score",
    "n_clusters",
    "enrich_data"
]


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        if not isinstance(columns, list):
            columns = [columns]
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


class HDBSCAN_w_transf(HDBSCAN):       
    def transform(self, X, y=None):
        return self.labels_


features = [
    'danceability',
    'energy',
    # 'loudness',
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

RANDOM_STATE = 42

def make_processing(w_dist=True, kwargs_umap=None, kwargs_hdbscan=None) -> Pipeline:

    if kwargs_umap is None:
        kwargs_umap = dict(
            n_neighbors=15,
            min_dist=0.1,
        )

    if kwargs_hdbscan is None:
        kwargs_hdbscan = dict(
            min_cluster_size=50,
            min_samples=None,
        )

    if w_dist:
        # default_metric = "precomputed"
        # dist_pipe = FunctionTransformer(
        #     lambda X: squareform(pdist(X, metric='mahalanobis'))
        # )
        default_metric = "mahalanobis"
        dist_pipe = "passthrough"
    else:
        default_metric = "euclidean"
        dist_pipe = "passthrough"

    mapper = UMAP(
        random_state=kwargs_umap.pop('random_state', RANDOM_STATE),
        metric=kwargs_umap.pop('metric', default_metric),
        n_epochs=kwargs_umap.pop('n_epochs', 200),
        verbose=kwargs_umap.pop('verbose', False),
        **kwargs_umap
    )

    clusterer = HDBSCAN(
        # approx_min_span_tree=kwargs_hdbscan.pop('approx_min_span_tree', False),
        metric=kwargs_hdbscan.pop('metric', "euclidean"),
        core_dist_n_jobs=kwargs_hdbscan.pop('core_dist_n_jobs', 1),
        gen_min_span_tree=kwargs_hdbscan.pop('gen_min_span_tree', True),
        **kwargs_hdbscan
    )
    
    return Pipeline([
        ('feature_selection', FeatureSelector(features)),
        ('dist', dist_pipe),
        ('mapper', mapper),
        ('clusterer', clusterer)
    ])


def make_processing_parallel(w_dist=False, kwargs_umap=None, kwargs_hdbscan=None) -> Pipeline:

    if kwargs_umap is None:
        kwargs_umap = dict(
            n_neighbors=15,
            min_dist=0.1,
        )

    if kwargs_hdbscan is None:
        kwargs_hdbscan = dict(
            min_cluster_size=50,
            min_samples=None,
        )

    if w_dist:
        default_metric = "precomputed"
        dist_pipe = FunctionTransformer(
            lambda X: squareform(pdist(X, metric='mahalanobis'))
        )
    else:
        default_metric = "euclidean"
        dist_pipe = "passthrough"

    mapper = UMAP(
        random_state=kwargs_umap.pop('random_state', RANDOM_STATE),
        metric=kwargs_umap.pop('metric', default_metric),
        n_epochs=kwargs_umap.pop('n_epochs', 200),
        verbose=kwargs_umap.pop('verbose', False),
        **kwargs_umap
    )

    # #HACK for HDBSCAN: https://github.com/scikit-learn-contrib/hdbscan/issues/73
    # metric = DistanceMetric.get_metric('mahalanobis', V=df2.cov())

    clusterer = HDBSCAN_w_transf(
        # approx_min_span_tree=kwargs_hdbscan.pop('approx_min_span_tree', False),
        metric=kwargs_hdbscan.pop('metric', "euclidean"),
        core_dist_n_jobs=kwargs_hdbscan.pop('core_dist_n_jobs', 1),
        gen_min_span_tree=kwargs_hdbscan.pop('gen_min_span_tree', True),
        **kwargs_hdbscan
    )

    # def transform(self, X, y=None):
    #     return self.labels_
    # setattr(clusterer, 'transform', transform)

    return Pipeline([
            ('feature_selection', FeatureSelector(features)),
            ('transf', FeatureUnion([
                ('mapper', Pipeline([('dist', dist_pipe), ('mapper', mapper)])),
                ('clusterer', clusterer)
            ]))
        ])


def find_clusterer(pipe):
    if "clusterer" in pipe.named_steps.keys():
        clusterer = pipe["clusterer"]
    else: #parallel
        named_steps = {t[0]:t[1] for t in pipe['transf'].transformer_list}
        clusterer = named_steps["clusterer"]
    return clusterer

def dbcv(pipe:Pipeline, X=None, y=None):
    clusterer = find_clusterer(pipe)
    score = clusterer.relative_validity_
    # HACK: for skopt.BayesianOptim, to avoid same results.
    if score == 0:
        return np.random.random_integers(10) * 1e-6
    return score


def n_clusters(pipe:Pipeline, X=None, y=None):
    clusterer = find_clusterer(pipe)
    return np.unique(clusterer.labels_).shape[0]


def dbcv_normalized(pipe:Pipeline, X=None, y=None):
    return dbcv(pipe, X, y) / np.log1p(n_clusters(pipe, X, y))


def dbcv_mult_n_clusters(pipe:Pipeline, X=None, y=None):
    return dbcv(pipe, X, y) * np.log1p(n_clusters(pipe, X, y))


def validity_score(pipe, X, y=None):
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
    return validity_index(X_map, clusterer.labels_, metric=clusterer.metric)


def dbcv_validity_score(pipe, X, y=None):
    return dbcv(pipe, X, y=None) * validity_score(pipe, X, y=None)


def abs_dbcv_validity_score(pipe, X, y=None):
    return dbcv(pipe, X, y=None) * abs(validity_score(pipe, X, y=None))


def make_default_search_param_spaces(n_samples=1000, n_search=3):
    return {
        'mapper__min_dist': np.geomspace(1e-3, 0.99, n_search),
        'mapper__n_neighbors': [int(x) for x in np.geomspace(5, n_samples, n_search)],
        'clusterer__min_cluster_size': [int(x) for x in np.geomspace(5, n_samples, n_search)],
        'clusterer__min_samples': [int(x) for x in np.geomspace(5, n_samples, n_search)],
    }


def make_default_optim_param_spaces(n_samples=1000, prior='log-uniform'):
    return {
        'mapper__min_dist': Real(1e-3, 0.99, prior=prior),
        'mapper__n_neighbors': Integer(5, n_samples, prior=prior),
        'clusterer__min_cluster_size': Integer(5, n_samples, prior=prior),
        'clusterer__min_samples': Integer(5, n_samples, prior=prior),
    }


def make_default_optim_param_spaces_parallel(n_samples=1000, prior='log-uniform'):
    return {
        'transf__clusterer__min_cluster_size': Integer(5, n_samples, prior=prior),
        'transf__clusterer__min_samples': Integer(5, n_samples, prior=prior),
    }


def make_search(
    proc=None,
    param_spaces=None,
    scoring=None,
    **kwargs
) -> GridSearchCV:

    if proc is None:
        proc = make_processing()

    param_spaces = 3 if param_spaces is None else param_spaces

    if isinstance(param_spaces, int):
        param_spaces = make_default_search_param_spaces(n_search=param_spaces)

    if scoring is None:
        scoring = {
            'DBCV':dbcv,
            'n_clusters':n_clusters,
            'validity':validity_score
        }
    
    gscv = GridSearchCV(
        proc,
        param_spaces,
        cv=[(slice(None), slice(None))],
        scoring=kwargs.pop('scoring', scoring),
        verbose=kwargs.pop('verbose', 2),
        return_train_score=kwargs.pop('return_train_score', False),
        refit=kwargs.pop('refit', False),
        n_jobs=kwargs.pop('n_jobs', 4),
        **kwargs
    )

    return gscv


def make_optimization(
    proc=None,
    search_spaces=None,
    **kwargs
) -> BayesSearchCV:

    if proc is None:
        proc = make_processing()

    if search_spaces is None:
        search_spaces = make_default_optim_param_spaces()

    bscv = BayesSearchCV(
        proc,
        search_spaces,
        cv=[(slice(None), slice(None))],
        scoring=kwargs.pop('scoring', abs_dbcv_validity_score),
        n_iter=kwargs.pop('n_iter', 200),
        verbose=kwargs.pop('verbose', 2),
        return_train_score=kwargs.pop('return_train_score', False),
        refit=kwargs.pop('refit', False),
        random_state=kwargs.pop('random_state', RANDOM_STATE),
        # optimizer_kwargs=kwargs.pop('optimizer_kwargs', {"initial_point_generator":'lhs'}),
        n_points=kwargs.pop('n_points', 8),
        n_jobs=kwargs.pop('n_jobs', 8),
        **kwargs
    )

    return bscv


def get_n_best_params(search, col_test_score='mean_test_score', n_best=5):
    params = pd.DataFrame(search.cv_results_)\
        .sort_values(col_test_score, ascending=False)\
        ['params']\
        .head(n_best)\
        .to_list()
    return [{**p} for p in params]


def analysis_proc(proc, params, df):
    results = []

    serial = "clusterer" in proc.named_steps.keys()

    for i, param in enumerate(params):
        # reset relative_validity to force re-calculcation
        if serial:
            proc["clusterer"]._relative_validity = None
            clusterer = proc["clusterer"]
        else: #parallel
            # named_steps = {t[0]:t[1] for t in proc['transf'].transformer_list}
            proc['transf'].transformer_list[1][1]._relative_validity = None
            clusterer = proc['transf'].transformer_list[1][1]

        proc = proc.set_params(**param).fit(df)

        X_map = df
        for name, estimator in proc.steps[:-1]:
            if isinstance(estimator, str) or estimator is None:
                continue
            X_map = estimator.transform(X_map)
        if not serial:
            X_map = proc['transf'].transformer_list[0][1].transform(X_map)

        result = {
            "x_map":X_map[:, 0],
            "y_map":X_map[:, 1],
            "cluster":clusterer.labels_,
            "dbcv":clusterer.relative_validity_,
            "n_clusters":np.unique(clusterer.labels_).shape[0],
            "validity_index":validity_index(X_map.astype(np.float64), clusterer.labels_)
        }
        results.append(result)
    
    return results


def analysis_plot(results, figsize=(12, 40)):
    fig, axs = plt.subplots(len(results), 1, figsize=figsize)

    for i , result in enumerate(results):
        ax = axs[i]
        m = ax.scatter(
            x=result['x_map'],
            y=result['y_map'],
            c=result['cluster'],
            cmap='Set2'
        )
        plt.colorbar(m, ax=ax)
        title = f'parameter {i}: dbcv: {result["dbcv"]:.3f} - n_clusters: {result["n_clusters"]} - validity_index: {result["validity_index"]:.3f}'
        ax.set_title(title)
    
    return fig


def analysis_plot_pipe(
    search,
    df,
    col_test_score='mean_test_score',
    n_best=5,
    figsize=(12, 40),
):
    params = get_n_best_params(search, col_test_score, n_best)
    results = analysis_proc(search.estimator, params, df)
    return params, analysis_plot(results, figsize=figsize)


def enrich_data(sp, df) -> pd.DataFrame:
    df = enrich_audiofeature(df, sp, col="track.id")
    df, genre = add_genres(df, sp)
    df.columns = df.columns.str.replace('track.id.','', regex=False).tolist()
    return df


def get_output(pipe, df):

    clusterer = find_clusterer(pipe)
    X_map = df.copy()
    for name, estimator in pipe.steps[:-1]:
        if isinstance(estimator, str) or estimator is None:
            continue
        X_map = estimator.transform(X_map)
        if isinstance(X_map, pd.DataFrame):
            X_map = X_map.to_numpy()
        else:
            X_map = np.float64(X_map)
            
    return pd.DataFrame({
        'cluster':clusterer.labels_,
        'x_map':X_map[:,0],
        'y_map':X_map[:,1],
        "dbcv":clusterer.relative_validity_,
        "n_clusters":np.unique(clusterer.labels_).shape[0],
        "validity_index":validity_index(X_map.astype(np.float64), clusterer.labels_)
    })


def plot_proj_cluster(result, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    m = ax.scatter(
        x=result['x_map'],
        y=result['y_map'],
        c=result['cluster'],
        cmap='Set2'
    )
    plt.colorbar(m, ax=ax)
    title = f'dbcv: {result.iloc[0]["dbcv"]:.3f} - n_clusters: {result.iloc[0]["n_clusters"]} - validity_index: {result.iloc[0]["validity_index"]:.3f}'
    ax.set_title(title)
    
    return ax