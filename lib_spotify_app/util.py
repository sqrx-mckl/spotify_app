import numpy as np
from pathlib import Path
import spotipy
import json
import pandas as pd
from typing import Dict, List, Union

from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import yule
from scipy.cluster.hierarchy import (
    fcluster, dendrogram, linkage, cut_tree, leaders)
from sklearn.cluster import OPTICS

import matplotlib.pyplot as plt
from IPython.display import display

import re
from itertools import chain
from collections import Counter
from statistics import mode

from .adapter_spotipy_api import SpotipyApi

import plotly.graph_objects as go

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


def normalize_request(_request)->pd.DataFrame:
    """
    transform the output of a request into a DataFrame
    
    Parameters
    ----------
    request : Dict?
        result of a request
    
    Returns
    -------
    pd.DataFrame
        transformed result of the request which contained nested dictionnary
    """
    # some request gives back a strange dict with key the name of the
    # request and values the lists output
    if isinstance(_request, dict) and 'items' in _request.keys():
        request = _request['items']
    elif isinstance(_request, dict) \
        and len(_request.keys()) == 1 \
        and isinstance(_request[list(_request.keys())[0]], list):
        request = _request[list(_request.keys())[0]]
    else:
        request = _request

    # if there is multilple request inside the request (like a list). The 
    # output is a list, else is a dict
    if isinstance(request, list):
        df_list = [pd.json_normalize(json_list2dict(r)) for r in request]
        df = pd.concat(df_list).reset_index()
    elif isinstance(request, dict):
        df = pd.json_normalize(json_list2dict(request))
    
    return df


def _enrich_by_feature(ser:pd.Series, f, w:int)->pd.DataFrame:
    """
    Helper function to retrieve the enriched data for enrich_df_by_feature
    
    Parameters
    ----------
    ser : pd.Series
        Initial Series to use for enrichment
    w : int
        Size of the rolling window (to request multiple rows at a time)
    f : function
        Function to use to enrich the data
    
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame
    """

    # Get unique set of values
    ser_un:pd.Series = pd.Series(ser.unique())

    # Get request group
    window_groups = [x // w for x in range(len(ser_un))]

    # do the request, normalize it and set as index the initial serie
    dfe = ser_un.groupby(window_groups)\
                .apply(lambda x: normalize_request(f(x)))\
                .set_index(ser_un)

    # "map" the index to the full initial index
    return dfe.loc[ser.to_list()]

def enrich_df_by_feature(df:pd.DataFrame, col:str, f, w:int)->pd.DataFrame:
    """
    Enrich the dataframe by requesting information
    The request is done via a function which is called with a rolling window.
    Use the following command to join your initial DataFrame with the enriched
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be enriched
    col : str
        Initial column to use for enrichment
    w : int
        Size of the rolling window (to request multiple rows at a time)
    f : function
        Function to use to enrich the data
    
    Returns
    -------
    pd.DataFrame
        [description]
    """

    df_enriched = _enrich_by_feature(df[col], f=f, w=w)
    df_enriched = df_enriched.add_prefix(f'{col}.')

    return df.join(df_enriched, on=col)
    
def enrich_audiofeature(df:pd.DataFrame,
                        adapter:SpotipyApi,
                        col:str='id')->pd.DataFrame:
    return enrich_df_by_feature(df,
                                col=col,
                                f=adapter.sp.audio_features,
                                w=100)


def plotly_categorical_scatter(df, x:str, y:str, hue:str, size:str, text:str, link:str)->go.FigureWidget:

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

def mask_outlier_iqr(x:pd.Series)->pd.DataFrame:
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    return pd.DataFrame({'high': x > q3 + 1.5*iqr,
                          'low': x < q1 - 1.5*iqr})