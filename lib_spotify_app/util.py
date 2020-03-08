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

class adapter_spotipy_api:
    """
    class to adapter spotipy for our usage and this specific configuration
    The authorization flow used here is authorization code:
    https://developer.spotify.com/documentation/general/guides/authorization-guide/#authorization-code-flow
    
    Parameters
    ----------
    credential_fp : Path
        path to the credential JSON file
        This JSON file must contains the following field:
            * username - the Spotify username
            * client_id - the client id of your app
            * client_secret - the client secret of your app
    scope : List[str]
        list of scope, see:
        https://developer.spotify.com/documentation/general/guides/scopes/

    Attributes
    ----------
    credential_fp
    scope
    cache_path: to save the token in a file automatically by Spotipy
    token_code: the token code
    sp: the Spotipy session

    """
    
    def __init__(self, credential_fp:Path, scope:List[str], cache_path:Path):
        self.scope = scope
        self.credential_fp = credential_fp
        
        with open(credential_fp) as file:
            self.credential = json.load(file)
        # save user token in ".cache-<username>" file at user defined location
        self.cache_path = Path(cache_path,
                                f'.cache-{self.credential["username"]}')

        # from "get_token"
        self.token_code = None
        # from "open_session"
        self.sp = None


    def refresh_token(self):
        pass

    def get_token(self):
        # token need to be refreshed
        try:
            self.token_code = spotipy.util.prompt_for_user_token(
                username=self.credential["username"],
                client_id=self.credential["client_id"],
                client_secret=self.credential["client_secret"],
                redirect_uri="http://localhost/",
                scope=scope,
                cache_path=self.cache_path
            )
        except:
            print("error with token retrieval")
            raise

        self.credential["token"] = {'code':self.token_code,
                                    'scope':self.scope}
        with open(self.credential_fp, 'w') as file:
            json.dump(self.credential, file)

    def open_session(self):
        self.sp = spotipy.Spotify(auth=self.credential['token']['code'])

    def query_liked_songs(self, tracks_count:int=-1, limit:int=50):
        result_liked_songs = None
        offset = 0

        while True:
            limit_temp = min([limit, tracks_count-offset])
            result_temp = self.sp.current_user_saved_tracks(
                limit=limit_temp,
                offset=offset
            )

            # append the dictionnary output
            if result_liked_songs is None:
                result_liked_songs = result_temp
            else:
                result_liked_songs['items'] += result_temp['items']

            offset = offset + limit

            # check condition if there is one
            if tracks_count > 0 and offset > tracks_count:
                break

        return result_liked_songs


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


def normalize_request(request_output)->pd.DataFrame:
    """
    transform the output of a request into a DataFrame
    
    Parameters
    ----------
    request_output : [type]
        [description]
    
    Returns
    -------
    pd.DataFrame
        [description]
    """
    # some request gives back a strange dict with key the name of the
    # request and values the lists output
    if isinstance(request_output, dict):
        _request_output = list(request_output.values())[0]
    else:
        _request_output = request_output

    # if there is multilple request inside the request (like a list). The 
    # output is a list, else is a dict
    if isinstance(request_output, list):
        df_list = [pd.json_normalize(json_list2dict(s)) for s in request_output]
        df = pd.concat(df_list).reset_index()
    elif isinstance(request_output, dict):
        df = pd.json_normalize(json_list2dict(r))
    
    return df


def enrich_by_feature(df:pd.DataFrame, col_artist:str, f):
    """
    Enrich the dataframe by requesting information.
    The request is done via a function which is called with a rolling window.
    
    Parameters
    ----------
    df : pd.DataFrame
        Initial DataFrame to enrich
    col : str
        Column to use for enrichment
    w : int
        Size of the rolling window (to request multiple rows at a time)
    
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame
    """

    window_groups = range(len(df)) // w

    dfe = df[[col]]\
        .groupby(window_groups)[col]\
        .apply(lambda x: normalize_request(f(x)))\
        .add_prefix(f'{col}.enrich.')
                            
    dfe = dfe.reset_index().set_index(df[col])

    return df.join(dfe, on=col)


class facade_enrich_artist_genre:
    
    def __init__(self, artists:pd.Series):
        """
        Class which handle the artists data and most particurarly the genre information from the artist.
        https://spotipy.readthedocs.io/en/2.9.0/#spotipy.client.Spotify.recommendation_genre_seeds

        This class take care of the processing of the genres:
            * cleaning the "useless" genre (<country_name>_indie)
            * remove outliers
            * cluster into "super_genre"
            * name the "super_genre"
            * enrich initial data with the newly added "super_genre"

        Clustering is done by 2 algorithms:
            * DBSCAN with OPTICS to detect the outliers
            * hierarchical clustering to create the "super-genre"

        The metric used for the genre combination is the Yule distance:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.yule.html
        It is a binary array disambleance distance. It uses the amount of time a True is encountered at the same index for both arrays
        
        Parameters
        ----------
        artists : pd.Series
            artists data to be transformed
        """
        self.data = artists
        self._mlb = MultiLabelBinarizer()
        
        # DataFrame with genre as a column and each row is an artist
        self.df_genre:pd.DataFrame = pd.DataFrame(
            self._mlb.fit_transform(self.data),
            columns=self._mlb.classes_,
            index=self.data.index
        )

    self cluster_genre(self, method:str='average'):


    def clean_useless_genre(self):
        """
        spotify contains strange genre such as "alabama_indie" which are not useful for our purpose. As such this method get rids of all of them
        """
        mask = self.df_genre.columns.str.contains(r'genre_\w+ indie')
        self.df_genre = self.df_genre.loc[:,~mask]

    