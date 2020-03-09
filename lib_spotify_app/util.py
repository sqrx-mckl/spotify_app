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


def normalize_request(request)->pd.DataFrame:
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
    if isinstance(request, dict):
        _request = list(request.values())[0]
    else:
        _request = request

    # if there is multilple request inside the request (like a list). The 
    # output is a list, else is a dict
    if isinstance(request, list):
        df_list = [pd.json_normalize(json_list2dict(r)) for r in _request]
        df = pd.concat(df_list).reset_index()
    elif isinstance(request, dict):
        df = pd.json_normalize(json_list2dict(_request))
    
    return df


def enrich_by_feature(ser:pd.Series, w:int, f)->pd.DataFrame:
    """
    Enrich the dataframe by requesting information.
    The request is done via a function which is called with a rolling window.
    Use the following command to join your initial DataFrame with the enriched data:
    "df.join(dfe, on=col)"

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

    window_groups = range(len(ser)) // w

    dfe = ser.groupby(window_groups)\
             .apply(lambda x: normalize_request(f(x)))\
             .add_prefix(f'{ser.name}.enrich.')\
             .set_index(ser)

    return dfe

