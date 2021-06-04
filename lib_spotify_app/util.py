import numpy as np
from pathlib import Path
import spotipy
import json
import pandas as pd
from typing import Dict, List, Union

import matplotlib.pyplot as plt
from IPython.display import display

import re
from itertools import chain
from collections import Counter
from statistics import mode

import plotly.graph_objects as go
import plotly.express as px

def plotly_scatter(df, cluster_col='cluster'):
    df['hover_text'] = df.apply(
        lambda x:
            f'<br>'+
            f'Full Song: <a href="{x["track.external_urls.spotify"]}">Play</a><br>' +
            f'Album: {x["track.album.name"]}<br>' +
            # f'Genres: {x["artists.genres"]}<br>'+
            # f'Super Genres: {x["artists.supergenres"]}<br>' +
            f'Super Genre 1: {x["artists.supergenre_1"]}<br>',
        axis=1
    )

    df['size'] = df['track.popularity'].apply(lambda x: np.log(x+1))

    fig = px.scatter(
        df,
        x='x_map',
        y='y_map',
        hover_name='fullname',
        hover_data=['hover_text'],
        text=df['track.preview_url'].map(lambda x: f'<a href="{x}">Play</a>'),
        size='size',
        color=df[cluster_col].astype(str),
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title=f'my songs per {cluster_col}'
    )

    return fig


def plotly_categorical_scatter(
    df,
    x:str,
    y:str,
    hue:str,
    size:str,
    text:str,
    link:str
) -> go.FigureWidget:

    import webbrowser

    def click_event(trace, points, state):
        print('test')
        display(points.customdata)
        [webbrowser.open(point.customdata) for point in points]

    fig = go.FigureWidget()

    for group_name, df_group in df.groupby(hue):
        fig.add_trace(go.Scattergl(
            x=df_group[x],
            y=df_group[y],
            name=group_name,
            text=df_group[text],
            marker_size=df_group[size],
            customdata=df_group[link],
            ids=df_group.index.to_list(),
        ))
        
    # Tune marker appearance and layout
    fig.update_traces(
        mode='markers', 
        hoverinfo='text',
        marker=dict(sizeref=2.*max(df[size])/(5.**2),
                    line_width=0)
    )
    fig.for_each_trace(
            lambda trace: trace.on_click(click_event, append=True)
        )
    return fig


def concatenate_col(df:pd.DataFrame, pattern:str) -> pd.Series:
    """
    Concatenate the columns of a pd.DataFrame into a single pd.Series.
    The values in the new pd.Series will be a list of the values in the row of df.

    Parameters
    ----------
    df : pd.DataFrame
        initial dataframe to concatenate all columns per row

    Returns
    -------
    pd.Series[List]
        the value in each row is a list of the values in each row of df
    """

    return df.filter(regex=pattern)\
             .apply(lambda x: x.dropna().to_list(), axis=1)

def proj_and_cluster(df_feat, mdl_proj, mdl_cluster) -> pd.DataFrame:
    """
    Apply projection model (t-sne or u-map for example) and a cluster model (hdbscan for example). Creates a DataFrame with following columns:
    * proj_x, proj_y: the projections 
    * clusters: the clusters as text

    Parameters
    ----------
    df_feat : [type]
        [description]
    mdl_proj : [type]
        [description]
    mdl_cluster : [type]
        [description]

    Returns
    -------
    pd.DataFrame
        [description]
    """
    df = pd.DataFrame(
        mdl_proj.fit_transform(df_feat),
        index=df_feat.index,
        columns=['proj_x', 'proj_y']
    )

    df['clusters'] = mdl_cluster.fit_predict(df)
    df['clusters'] = df['clusters'].apply(lambda x: f'c{x}')

    return df

