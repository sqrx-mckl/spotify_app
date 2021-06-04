## imports

from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import spotipy
from tqdm import tqdm
from sklearn.pipeline import Pipeline
import plotly.express as px
import hiplot as hip
import streamlit as st
from lib_spotify_app.model import (
    dbcv,
    make_processing,
    make_optimization,
    make_default_optim_param_spaces,
    analysis_plot_pipe,
    enrich_data,
    get_output
)
from lib_spotify_app.util import plotly_scatter
from lib_spotify_app.api_adapter import (
    make_spotify_playlist,
    setup_spotipy,
    query_liked_songs,
    get_playlist
)

# import ptvsd
# ptvsd.enable_attach(address=('localhost', 5678))
# ptvsd.wait_for_attach() # Only include this line if you always wan't to attach the debugger

## attributes
config = {
    "credential_fp":"private/spotify_credential.json",
    "cache_path":"private/credential_spotipy",
}

# Playlist ids for example
# '37i9dQZF1DXca8AyWK6Y7g', # young and free
# '37i9dQZF1DWTBN71pVn2Ej', # Alternative Noise
# '37i9dQZF1DX3IplhwNexYg', # pulp

text_example = """playlist:37i9dQZF1DXca8AyWK6Y7g
playlist:37i9dQZF1DWTBN71pVn2Ej
playlist:37i9dQZF1DX3IplhwNexYg"""
# user:squarex"""


st.set_page_config(layout="wide")


@st.cache(allow_output_mutation=True)
def get_spotipy_session(username:Optional[str]=None, config=config):
    if username is None:
        scope = None
    else:
        scope=[
            'user-library-read',
            'user-top-read',
            'playlist-modify-public',
            'playlist-modify-private'
        ]

    return setup_spotipy(
        client_credential_fp=config['credential_fp'],
        username=username,
        scope=scope,
        cache_path=config['cache_path'],
    )


sp_default = get_spotipy_session() # for non-logged-user data retrieval

@st.cache(allow_output_mutation=True)
def load_data(
    sp:spotipy.Spotify=sp_default,
    playlist_id=None,
    user_id=None,
)->Tuple[str, pd.DataFrame]:

    if playlist_id is not None and user_id is None:
        name, df = get_playlist(sp, playlist_id, get_name=True)
    elif user_id is not None and playlist_id is None:
        sp_user = get_spotipy_session(username=user_id)
        df = query_liked_songs(sp_user)
        name = user_id

    return name, df


@st.cache(allow_output_mutation=True)
def proc_data(dfs:List[pd.DataFrame], sp=sp_default) -> pd.DataFrame:

    df = pd.concat(dfs, ignore_index=True)
    df = enrich_data(sp, df)
    df = df.drop_duplicates(subset='track.id')

    df['fullname'] = df.apply(
        lambda x: f'{x["track.name"]}, {x["track.artists.0.name"]}<br>',
        axis=1
    )

    return df


@st.cache(allow_output_mutation=True)
def ml_data(df:pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame]:

    MAGIC_ML = {
        'clusterer__min_cluster_size': int(2*np.sqrt(df.shape[0])), # 100
        'clusterer__min_samples': int(np.sqrt(df.shape[0])), # 100
        'mapper__min_dist': 0.001,
        'mapper__n_neighbors': int(np.sqrt(df.shape[0])), # 100
    }

    proc = make_processing(w_dist=True)
    proc = proc.set_params(**MAGIC_ML)
    proc.fit(df)
    
    df_output = get_output(proc, df)
    df_out = pd.concat([df.reset_index(drop=True), df_output], axis=1)

    return proc, df_out


def calculate_data(input_text:str):
    input_list = input_text.split('\n')
    input_list = [(x.split(':')[0], x.split(':')[1]) for x in input_list]
    assert all([k!='' and v!='' for (k, v) in input_list])
    dfs = []
    for (k, v) in input_list:
        if k == 'playlist':
            name, df = load_data(playlist_id=v)
        elif k=='user':
            name, df = load_data(user_id=v)
        else:
            AttributeError('Wrong ')
        df['input_name'] = name
        dfs.append(df)
    df = proc_data(dfs)
    proc, df = ml_data(df)
    return proc, df


def show_data(df, group_col='cluster'):
    return df\
        .sort_values('track.popularity')\
        .groupby(group_col)[[
            group_col,
            'fullname',
            'track.external_urls.spotify',
            'track.preview_url',
            'artists.supergenre_1',
            'artists.genres',
            'track.popularity',
        ]].head(5)


hiplot_features = [
    'danceability',
    'energy',
    # 'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
]

groupby_dict = {
    'Playlist': 'input_name',
    'Genres': 'artist.genres',
    'SuperGenres': 'artist.supergenre_1',
    'Cluster': 'cluster'
}

# Dashboard

# View creation
st.title('My Spotify Playlist Dashboard')

left_column, right_column = st.beta_columns(2)

with st.sidebar:
    st.title('Data Collection')
    st.markdown("""Please enter the playlists you would like to analyse.
Use the format specified in example:
- `user` means that the user-liked-songs playlist will be retrieved
- `playlist` means that the public playlist will be retrieved""")
    input_text = st.text_area("List of Playlists", value=text_example)
    is_ready = st.button("Validate")

if is_ready:
    proc, data = calculate_data(input_text)

    groupby_choice = st.selectbox('Group By',
                                  ('Playlist', 'Genres', 'SuperGenres', 'Clusters'))
    groupby_col = groupby_dict[groupby_choice]
    st.write('You selected to group by:', groupby_choice)

    st.subheader(groupby_choice)
    st.plotly_chart(plotly_scatter(data, cluster_col=groupby_col))
    st.dataframe(data=show_data(data, groupby_col).sort_values(groupby_col))

    xp = hip.Experiment.from_dataframe(data[hiplot_features])
    # filtered_uids, selected_uids = xp.to_streamlit(
    #     ret=["filtered_uids", "selected_uids"],
    #     key='feature_hiplot'
    # ).display()
    xp.to_streamlit(key='feature_hiplot').display()


#TODO: Hiplot refresh makes the view disapear
#TODO: optimize the cluster processing (with use of a button & a spinner to wait)
#TODO: VS Code - add debugger with ptvsd