#%% [markdown]
# First, let's setup all of this...

#%% Imports
import numpy as np
from pathlib import Path
import spotipy
import json
from pprint import pprint
import dill
import seaborn as sns
sns.set_style('whitegrid')

import lib_spotify_app

# import modin as pd
import pandas as pd
pd.set_option('max_columns', None)
from typing import Dict, List, Union

from copy import deepcopy

# User configuration file, json with specifics key
credential_fp = Path(r'private/spotify_credential.json')

# List of interesting scopes
scope = ' '.join([
    'user-library-read',
    'user-top-read',
])

#%% Let's retrieve token authorization access
with open(credential_fp) as file:
    credential = json.load(file)

# token need to be refreshed
try:
    token_code = spotipy.util.prompt_for_user_token(
        username=credential["username"],
        client_id=credential["client_id"],
        client_secret=credential["client_secret"],
        redirect_uri="http://localhost/",
        scope=scope,
        cache_path=Path(f'private/.cache-{credential["username"]}')
    )
    credential["token"] = {'code':token_code,
                            'scope':scope}
    with open(credential_fp, 'w') as file:
        json.dump(credential, file)
except:
    print("error with token retrieval")
    raise

#%% [markdown]
# Now we start

#%% Do some requests...
sp = spotipy.Spotify(auth=credential['token']['code'])

#%% Run simple queries

limit = 50
result_top_tracks = sp.current_user_top_tracks(limit=limit)
result_top_artists = sp.current_user_top_artists(limit=limit)

#%% Checkpoint save
dill.dump_session(Path('.env_dump', 'spotify.db'))

#%% Checkpoint load
dill.load_session(Path('.env_dump', 'spotify.db'))

#%% Copy the results
top_artists = deepcopy(result_top_artists['items'])
top_tracks = deepcopy(result_top_tracks['items'])

#%% test
pprint(top_tracks[0].keys())
pprint(top_artists[0].keys())
pprint(top_tracks[0])
pd.json_normalize(top_tracks)

#%%[markdown]
# Ok, so now let's understand the results.
# What could be an issue once normalized or read...

artists_example = [tt for tt in top_tracks if len(tt['artists']) > 1]
pprint(artists_example)


#%%[markdown]
# So the problem is that there can be multiple artists, to take care of this, 
# artists, instead of a list, should be a dictionnary
# %% normalize json

def json_list2dict(d:Dict)->Dict:
    """
    Loop through all fields, and once it meet a list, it convert into a dict.
    The converted output contains the index of the list as a key.
    Conversion is done deeply to last level.
    
    Parameters
    ----------
    d : Dict
        initial dict to convert
    
    Returns
    -------
    Dict
        converted dict
    """

    for key, val in d.items():
        # convert list 2 dict with key as the index if it contains a container
        if isinstance(val, list) \
        and len(val) > 0 \
        and isinstance(val[0], (list, dict)):
            val = {str(k):v for k, v in enumerate(val)}
        # recursion (even for the newly converted list)
        if isinstance(val, dict):
            val = json_list2dict(val)
        d[key] = val

    return d

top_tracks = [json_list2dict(tt) for tt in top_tracks]

# Convert to DataFrame
df_topartists = pd.json_normalize([json_list2dict(tt) for tt in top_artists])
df_toptracks = pd.json_normalize(top_tracks)

#%% Check
pprint(df_toptracks.info())
df_toptracks.sample(5)

#%% Enrichment
# request the artist (full info)

def enrich_loop(df:pd.DataFrame, col:str, f, w:int)->pd.DataFrame:
    """
    Enrich the dataframe by requesting information.
    The request is done via a function which is called with a rolling window.
    
    Parameters
    ----------
    df : pd.DataFrame
        Initial DataFrame to enrich
    col : str
        Column to use for enrichment
    f : function
        Function to use to do the request, input is a pd.Series with "w" rows
        from the column "col"
    w : int
        Size of the rolling window (to request multiple rows at a time)
    
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame
    """

    def normalized_request(x:pd.DataFrame)->pd.DataFrame:
        r = f(x)
        # some request gives back a strange dict with key the name of the
        # request and values the lists output
        if isinstance(r, dict):
            r = list(r.values())[0]
        # now I normalize the json output of the request (as done before)
        if isinstance(r, list):
            df_list = [pd.json_normalize(json_list2dict(s)) for s in r]
            df = pd.concat(df_list).reset_index()
        elif isinstance(r, dict):
            df = pd.json_normalize(json_list2dict(r))
        return df

    dfe = df[[col]].groupby(range(len(df)) // w)[col]\
                            .apply(normalized_request)\
                            .add_prefix(f'{col}.enrich.')
                            
    dfe = dfe.reset_index().set_index(df[col])
    
    #%% gives back "df" as it was (no inplace)
    df = df.drop('window', axis=1)

    return df.join(dfe, on=col)

#%%
df_toptracks = df_toptracks.pipe(enrich_loop,
                                 col='artists.0.id',
                                 f=sp.artists,
                                 w=50)\
                           .pipe(enrich_loop,
                                 col='id',
                                 f=sp.audio_features,
                                 w=100)

#%% Checkpoint save

df_toptracks.to_pickle(Path(r'private/toptracks.pickle'))

#%% Checkpoint load

df_toptracks = pd.read_pickle(Path(r'private/toptracks.pickle'))

#%% Plot all available column for 1 row
with pd.option_context('display.max_rows', None):
    pprint(df_toptracks.iloc[0,:])
#%%[markdown]
# Let's retrieve the most useful column for our analysis like:
# * related to music features
# * to classify it (name, etc...)
# * release date (could be related to a specific era)
#%% quick analysis

df:pd.DataFrame = df_toptracks[[
    'id',
    'name',
    'artists.0.id.enrich.name',
    'album.release_date',
    'artists.0.id.enrich.genres',
    'popularity',
    'duration_ms',
    'artists.0.id.enrich.popularity',
    'artists.0.id.enrich.followers.total',
    'id.enrich.danceability',
    'id.enrich.energy',
    'id.enrich.key',
    'id.enrich.loudness',
    'id.enrich.mode',
    'id.enrich.speechiness',
    'id.enrich.acousticness',
    'id.enrich.instrumentalness',
    'id.enrich.liveness',
    'id.enrich.valence',
    'id.enrich.tempo',
    'id.enrich.time_signature',
]]

df_plot:pd.DataFrame = df.drop([
    'id',
    'name',
    'artists.0.id.enrich.name',
    'album.release_date',
    'artists.0.id.enrich.genres'
], axis=1)

sns.set_context('notebook')
#%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 9) # (w, h)

df_plot.hist(sharey=True)

#%%[markdown]
# We will try a clustering algorithm and use the "genre" feature to check if
# the clustering succeedded or is failing totally.
# As such we will cluster with T-SNE and then do a deeper analysis of the 
# "genre" feature:
# https://spotipy.readthedocs.io/en/2.9.0/#spotipy.client.Spotify.recommendation_genre_seeds
# * how many unique genre there is
# * how many unique combination of genre there is
# * what is the distribution of genre and same for its combination
# * distribution of the number of genre per sample

#%% cluster via T-SNE

from sklearn.manifold import TSNE

tsne_pos = TSNE().fit_transform(df.drop([
    'id',
    'name',
    'artists.0.id.enrich.name',
    'album.release_date',
    'artists.0.id.enrich.genres',
    'popularity',
    'duration_ms',
    'artists.0.id.enrich.popularity',
    'artists.0.id.enrich.followers.total',
], axis=1))

df[['tsne_posx', 'tsne_posy']] = pd.DataFrame(tsne_pos)

#%% genre analysis - analysis of combination
df['genre'] = df['artists.0.id.enrich.genres'].map(lambda x: '_'.join(x))
df.genre.value_counts().plot(kind='barh')

#%% create genre matrix
# multi-label binarizer to apply first, column length = number of unique genre
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_bin = mlb.fit_transform(
    df['artists.0.id.enrich.genres']
)
df_genre:pd.DataFrame = pd.DataFrame(genre_bin,
                                    columns=mlb.classes_,
                                    index=df.index)\
                            .add_prefix('genre_')
df:pd.DataFrame = pd.concat([df,df_genre],axis=1)

#%% quick analysis
df_genre.sum().sort_values().plot(kind='barh')

#%% [markdown]
# first I delete the genre which are not relevant, such as:
# * those from specific US states such as "brooklyn_indie"

#%% clean

df_genre = df_genre.loc[:,~df_genre.columns.str.contains(r'genre_\w+ indie')]

#%%
# To analyse genre combination I use the Yule distance (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.yule.html) which is a binary array disambleance distance.
# It uses the amount of time a True is encountered at the same index for both arrays

#%% test yule distance
from scipy.spatial.distance import yule
sns.heatmap(df_genre.corr(method=yule),
            cmap='RdYlGn_r',
            xticklabels=False,
            yticklabels=True)

#%% Analyse genre combination
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
import matplotlib.pyplot as plt

def plot_clusters(method:str, df_genre:pd.DataFrame=df_genre):
    plt.figure()
    g = sns.clustermap(df_genre.transpose(),
                        method=method,
                        metric='yule', 
                        col_cluster=False,
                        yticklabels=True)
    g.fig.suptitle(method) 

    L = g.dendrogram_row.linkage

    plt.figure().suptitle(method)
    _ = dendrogram(L,
                    orientation='left',
                    labels=df_genre.columns.to_list())

    return L
#%%
L_single = plot_clusters('single')
L_complete = plot_clusters('complete')

L_weighted = plot_clusters('weighted')
L_average = plot_clusters('average')


#%% [markdown]
# Now the idea is to retrieve the clusters made by Seaborn to simplify the 
# genres in bigger groups/clusters and as such reduce the number of genre to 
# less than 10 and as such reduce the number of combined genre. Such 
# super-genres could be:
# * electronic
# * alt. rock pop
# * rock
# * other

#%%

from IPython.display import display

def get_cluster_df(cl, df_genre=df_genre)->pd.DataFrame:
    df_genre_cluster:pd.DataFrame = df_genre.columns.to_frame()\
                                            .reset_index(drop=True)\
                                            .assign(cluster=cl)\
                                            .pivot(columns='cluster')

    # reduce row size, only need a dropping list of genres
    df_genre_cluster = df_genre_cluster.apply(
        lambda x: x.sort_values().reset_index(drop=True)
    )
    df_genre_cluster = df_genre_cluster.dropna(how='all')

    # # delete cluster of size 1
    # df_genre_cluster = df_genre_cluster.dropna(axis=1, thresh=2)

    return df_genre_cluster

def get_cluster_from_linkage(Z, t, df_genre=df_genre)->pd.DataFrame:
    cl = fcluster(Z, t, criterion='distance')
    return get_cluster_df(cl, df_genre=df_genre)

def display_cluster_from_linkage(Z, t, method:str, df_genre=df_genre):
    with pd.option_context('display.max_rows', None):
        print(f'{method} - {t}')
        display(get_cluster_from_linkage(Z, t, df_genre=df_genre))

def display_cluster(cl, method:str, df_genre=df_genre):
    with pd.option_context('display.max_rows', None):
        print(method)
        display(get_cluster_df(cl, df_genre=df_genre))

#%%
display_cluster_from_linkage(L_average, 1.8, 'average')
display_cluster_from_linkage(L_average, 1.6, 'average')
display_cluster_from_linkage(L_weighted, 1.6, 'weighted')
display_cluster_from_linkage(L_weighted, 1.4, 'weighted')
display_cluster_from_linkage(L_single, 1.0, 'single')
display_cluster_from_linkage(L_single, 0.1, 'single')

#%% Chosen from this study
display_cluster_from_linkage(L_weighted, 1.4, 'weighted')

#%%[markdown]
# Missing:
# * give name to super-genre
# * transform the genres into an unique super-genre (which to chose ??)
# * Maybe there is a better algorithm to define the clusters and the minimum list of super-genres to only have 1 super-genres per row...
# test this
# test DBSCAN (maybe better ?)

#%% test DBSCAN

from sklearn.cluster import OPTICS

cl = OPTICS(
    max_eps=1,
    min_samples=2,# to avoid lonely genre, goal is to have super-genre
    metric='yule',
    cluster_method='dbscan',
    algorithm='brute',# only valid algorithm for this metric
    n_jobs=-2
)
cl_optics = cl.fit_predict(df_genre.transpose())

display_cluster(cl_optics, method='dbscan')

#%% test leaders (to find outlier with hierarchy cluster)

from scipy.cluster.hierarchy import cut_tree, leaders

display(fcluster(L_weighted, 1.99, criterion='distance'))
display(leaders(L_weighted, fcluster(L_weighted, 1.99, criterion='distance')))

# %% [markdown]
# Proposal:
# * use OPTIC-DBSCAN to create initial clusters
# * cluster -1 become "other"
# * use hierarchy clustering to cluster the big cluster "0"
# * merge both super-genre clusters
# * re-apply on each sample/artist (one artist could have 2 super-genre...)
# * check and plot T-SNE as such

# %% OPTIC + hierarchy cluster

df_genre_cluster_optic = get_cluster_df(cl_optics)
# genre_cluster_0 = df_genre_cluster_optic[0][0]
genre_cluster_out = df_genre_cluster_optic[0][-1].dropna()

# df_genre2 = df_genre[genre_cluster_0]
df_genre2 = df_genre.drop(genre_cluster_out, axis=1)

L_weighted2 = plot_clusters('weighted', df_genre=df_genre2)

display_cluster_from_linkage(
    L_weighted2,
    1.5,
    'weighted',
    df_genre=df_genre2
)

#%% [markdown]
# ok now that's not so bad, let's plot T-SNE (maybe too much genre but will be 
# interesting with all the liked songs)
#
# So in the end, seems more robust to cluster everything out of the outlier 
# (the small clusters are found again)
# 
# Now time to wrap this up and plot T-SNE !
# 
# And not forget to name the super-genre...
# 
# Biggest issue will be to fit the legend with the plot (should be easy...)

#%% concat all

df_genre_super = get_cluster_from_linkage(L_weighted2, 1.8, df_genre=df_genre2)

df_genre_super = df_genre_super[0].combine_first(
    genre_cluster_out.to_frame().rename_axis('cluster')
)
display(df_genre_super)

#%% name the super genre

import re
from itertools import chain
from collections import Counter
from statistics import mode

def split_word(x:str)->List[str]:
    """
    split a string as a list of words
    
    Arguments:
        x {str} -- string to split
    
    Returns:
        List[str] -- list of words (as strngs)
    """
    if x is np.nan:
        return []
    x = x.replace('genre_', '')
    return re.findall(r"[\w']+", x)

def robust_str_mode(x:List[str], sep:str='-')->str:
    """
    robust method to calculate the mode of a list of string, returns a concatenate string in case of equal number of counter
    
    Arguments:
        x {List[str]} -- list of string to calcualte the mode
        sep {str} -- string used to concatenate mutliple string in case of draw
    
    Returns:
        [str] -- mode word
    """
    try:
        return mode(x)
    except:
        vc = pd.Series(x).value_counts()
        x_to_join = vc.index[vc == vc.max()].values
        return '-'.join(x_to_join)

super_genre = df_genre_super\
    .applymap(lambda x: split_word(x))\
    .agg(lambda x: robust_str_mode(list(chain(*x.to_list()))))

df_genre_super.columns = super_genre

df_genre_super = df_genre_super.applymap(
    lambda x: str(x).replace('genre_', '') if isinstance(x, str) else x
)

df_genre.columns = df_genre.columns.str.replace('genre_', '')

#%%

# This new Series will contains the amount of time a super-genre is met, this will be his weight
genre_weight = df_genre.sum()
genre_super_weight = df_genre_super.agg(
    lambda x: genre_weight[x.dropna()].sum()
)
genre_weight = genre_weight.append(genre_super_weight)

display(genre_weight)
genre_super_weight.plot(kind='barh')

#%%

def convert_genre_df(df:pd.DataFrame, 
                    df_genre=df_genre,
                    genre_super_weight=genre_super_weight)->pd.Series:
    """
    just because I wrongly created this matrice and I need a series with just the genre and the super-genre as a value... Guess I need to refactor...
    
    Arguments:
        df {[type]} -- [description]
    """

    def find_super_genre_from_genre(genre):
        super_genre_list = df\
            .apply(lambda x: x.str.match(genre))\
            .any()\
            .loc[lambda x: x]\
            .index.values

        super_genre = genre_super_weight[super_genre_list].idxmax()

        return super_genre

    genre_conversion = df_genre.columns.to_series().apply(
        lambda x: find_super_genre_from_genre(x)
    )

    return genre_conversion

def enrich_super_genre(
    df:pd.DataFrame=df,
    df_genre:pd.DataFrame=df_genre_super,
    genre_col:str='artists.0.id.enrich.genres',
    genre_super_weight=genre_super_weight
)->pd.DataFrame:
    """
    Enrich a dataframe of songs with "super-genre" feature, super_genre DataFrame is needed for conversion
    
    Keyword Arguments:
        df {pd.DataFrame} -- song DataFrame (default: {df})
        df_genre {pd.DataFrame} -- DataFrame containing a dictionnary of genre clustered in super-genre (default: {df_genre_super})
    
    Returns:
        pd.DataFrame -- same as df with a newly added feature/column
    """
    
    conversion_serie = convert_genre_df(df_genre_super)
    
    s_super_genre = df[genre_col].apply(
        lambda x: genre_super_weight[conversion_serie[x].values].idxmax() if x else x
    )

    return df.assign(super_genre=s_super_genre)

# df.pipe((enrich_super_genre,'df'), df_genre=df_genre_super)
display(convert_genre_df(df_genre_super))
display(enrich_super_genre())

#%%[markdown]
# have to refactor the code with a "genre" class which manage the genre, super-genre and clustering...

#%% [markdown]
# Missing:
# * get super-cluster as new features
# * plot T-SNE with "hue" this new feature
# * repeat with "Likes songs"

#%% Plots TSNE classification

sns.scatterplot(
    data=df,
    x='tsne_posx', 
    y='tsne_posy'
)

#%%
# (apply sklearn.preprocessing.MultiLabelBinarizer on "genre")
#//TODO request the audio features

#%%[markdown]
# a new playlist with top5 songs of top50 artists could be useful (with taking 
# out the songs already liked)

#%% [markdown]
# # Liked Songs analysis
# when I tackle the main task: cluster the liked songs playlist
#%% run big query
# save_tracks_count = 500 # //TODO: to modify to query all songs
# result_liked_songs = None
# for offset in range(0, save_tracks_count, limit):
#     limit_temp = min([limit, save_tracks_count-offset])
#     result_temp = sp.current_user_saved_tracks(limit=limit_temp, \
#                                                 offset=offset)
#     if result_liked_songs == None:
#         result_liked_songs = result_temp
#     else:
#         result_liked_songs['items'] += result_temp['items']

# liked_songs = deepcopy(result_liked_songs['items'])
# df_likedsongs = pd.json_normalize([json_list2dict(tt) for tt in liked_songs])