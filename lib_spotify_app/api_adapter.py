from pathlib import Path
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
from spotipy.cache_handler import CacheHandler, CacheFileHandler
import json
from typing import Dict, List, Optional, Union
import pylast
import pandas as pd
import numpy as np
from functools import partial

__all__ = [
    "get_credential",
    "setup_spotipy",
    "setup_lastfm",
    "query_loop",
    "query_liked_songs",
    "normalize_request",
    "enrich_df_by_feature",
    "enrich_audiofeature",
    "make_spotify_playlist",
    "get_playlist"
]


def get_credential(credential_fp:Union[Path, str]) -> Dict:
    if isinstance(credential_fp, Path):
        credential_fp = str(credential_fp)
    with open(credential_fp) as file:
        credential = json.load(file)
    return credential


def get_cache(
    cache_path:Optional[str]=None,
    username:Optional[str]=None
) -> CacheFileHandler:

    if Path(cache_path).suffix == '' and username is None:
        cache_path = f"{cache_path}.cache"
    elif Path(cache_path).suffix == '':
        cache_path = f"{cache_path}.cache-{username}"

    return CacheFileHandler(cache_path=cache_path, username=username)


def setup_spotipy(
    client_credential_fp:Optional[Union[str, Path]]=None,
    username:Optional[str]=None,
    scope:Optional[List[str]]=None,
    cache_path:Optional[Union[str, Path]]=None,
) -> spotipy.Spotify:
    """
    Initialize an instance of the Spotify API without know-how of the setup internal.

    Parameters
    ----------
    cache_path : Optional[Union[str, Path]]
        If provided: Path to the cache folder for the authentification files
    scope : List[str]
        list of scope, see:
        https://developer.spotify.com/documentation/general/guides/scopes/
    username: Optional[str]:
        Spotify username (else will use a public session, no logged users)
    credential_fp : Optional[Union[str, Path]]
        If not provided, will use environment variable

    Returns
    -------
    spotipy.Spotify
        Spotipy API instance
    """

    scope = scope if scope is None else  ' '.join(scope)
    if isinstance(cache_path, Path):
        cache_path = str(cache_path)

    if client_credential_fp is None:
        credential = {"client_id":None, "client_secret":None} # use environment variable
    else:
        credential = get_credential(credential_fp=client_credential_fp)

    # if (username is None) and (scope is None):
    #     auth_manager = SpotifyClientCredentials()

    # to cache or not to cache the client id, client secret and username secret
    if cache_path is None:
        cache_handler = None
    else:
        cache_handler = get_cache(cache_path, username)

    # new method as per <https://github.com/plamere/spotipy/issues/263>
    auth_manager = SpotifyOAuth(
            client_id=credential["client_id"],
            client_secret=credential["client_secret"],
            redirect_uri="http://localhost:8080",
            scope=scope,
            cache_handler=cache_handler
        )

    return spotipy.Spotify(auth_manager=auth_manager)


def setup_lastfm(credential_fp) -> pylast.LastFMNetwork:
        """
        You have to have your own unique two values for API_KEY and API_SECRET
        Obtain yours from https://www.last.fm/api/account/create for Last.fm
        In order to perform a write operation you need to authenticate yourself
        """
        with open(credential_fp) as file:
            credential = json.load(file)

        # check that all credential necessary to launch the API are in JSON
        credential_keys = ['API_KEY', 'API_SECRET', 'username', 'password']
        is_json_ok = all([x in list(credential.keys())
                         for x in credential_keys])
        
        if not is_json_ok:
            raise EnvironmentError(
            f'{credential_fp} is incomplete, must contains:'+
            '\n-API_KEY\n-API_SECRET\n-username\n-password')

        # initiate an API session
        api = pylast.LastFMNetwork(
            api_key=credential['API_KEY'],
            api_secret=credential['API_SECRET'],
            username=credential['username'],
            password_hash=pylast.md5(credential['password'])
        )
        
        # get user information instance
        api.user = api.get_user(credential['username'])

        return api


def query_loop(
    query_function,
    tracks_count:int=None,
    limit:int=50,
    verbose=False
) -> pd.DataFrame:
    """
    Query in a loop.
    Spotify API limit the number of queried songs to 50 or 100 at a time.
    
    Parameters
    ----------
    tracks_count : int, optional
        quantity of tracks to query, chose None to query all tracks
        by default None
    limit : int, optional
        how many tracks to query at once, maximum is 50, by default 50
    
    Returns
    -------
    pd.DataFrame
        The results from a list of JSON request results to a DataFrame
    """

    # argument test
    if isinstance(query_function, partial):
        function_var = query_function.func.__code__.co_varnames
    else:
        function_var = query_function.__code__.co_varnames
    assert 'limit' in function_var, "query function is not a spotipy loop query"
    assert 'offset' in function_var, "query function is not a spotipy loop query"

    result = {'items':[]}

    while True:
        # cache the previous result
        result_item_temp = result['items']

        # new query with recursive arguments
        result = query_function(
            limit=limit,
            offset=len(result_item_temp)
        )

        # update info using query result
        if tracks_count is None:
            tracks_count = result['total']

        # append the dictionnary output
        result['items'] += result_item_temp

        if verbose:
            count_prev = len(result_item_temp)
            count_next = len(result['items'])
            print(f'Download: {count_prev} -> {count_next}')

        # update argument for last query adjustement
        limit = min([limit, tracks_count-len(result['items'])])

        if limit <= 0:
            break

    return normalize_request(result)

def query_liked_songs(
    sp:spotipy.Spotify,
    tracks_count:int=None,
    limit:int=50,
    verbose=False
) -> pd.DataFrame:
    """
    Query the current user liked songs in a loop.
    Spotify API limit the number of queried songs to 50 at a time.
    
    Parameters
    ----------
    tracks_count : int, optional
        quantity of tracks to query, chose None to query all tracks
        by default None
    limit : int, optional
        how many tracks to query at once, maximum is 50, by default 50
    
    Returns
    -------
    pd.DataFrame
        The results as in a list of JSON request results
    """

    return query_loop(
        sp.current_user_saved_tracks,
        tracks_count=tracks_count,
        limit=limit,
        verbose=verbose
    )


def normalize_request(_request) -> pd.DataFrame:
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
        df_list = []

        for r in request:
            # Spotify API returns some member of the request as "None"
            if r is None:
                # Trick to add NaN values at this row
                df_list.append(pd.json_normalize({}))
            else:
                df_list.append(pd.json_normalize(_json_list2dict(r)))

        if not df_list: # all is None
            raise ValueError("all values from request are None")
        
        df = pd.concat(df_list).reset_index(drop=True)
    
    elif isinstance(request, dict):
        df = pd.json_normalize(_json_list2dict(request))

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
    ser_unique = ser.drop_duplicates()

    # Get request group
    window_groups = [x // w for x in range(ser_unique.shape[0])]

    # do the request, normalize it and set as index the initial serie
    dfe:pd.DataFrame = ser_unique.groupby(window_groups)\
                                 .apply(lambda x: normalize_request(f(x)))\
                                 .set_index(ser_unique)

    # "map" the index to the duplicated initial index
    #return dfe.merge(ser, how='right', left_index=True, right_on=ser.name)
    return dfe


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

    #return df.merge(df_enriched, on=col, how="left")
    return df.merge(df_enriched, left_on=col, right_index=True, how='left')
    

def enrich_audiofeature(df:pd.DataFrame,
                        sp:spotipy.Spotify,
                        col:str='id')->pd.DataFrame:
    return enrich_df_by_feature(df,
                                col=col,
                                f=sp.audio_features,
                                w=100)


# Helper functions #

def _json_list2dict(d:Dict)->Dict:
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
            val = _json_list2dict(val)
        d[key] = val

    return d


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def make_spotify_playlist(
    name:str,
    sp:spotipy.Spotify,
    username:str,
    songs:pd.Series
) -> None:
    r_playlist = sp.user_playlist_create(username, name)

    for track_ids in chunks(songs.to_list(), 50):
        sp.playlist_add_items(r_playlist['id'], track_ids)

    # return boolean ?


def get_playlist(
    sp:spotipy.Spotify,
    playlist_uri:str,
    get_name:bool=False,
    tracks_count:Optional[int]=None,
    limit:int=100,
    verbose:bool=False
) -> pd.DataFrame:
    
    df = query_loop(
        partial(sp.playlist_tracks, playlist_id=playlist_uri),
        tracks_count=tracks_count,
        limit=limit,
        verbose=verbose
    )
    
    if get_name:
        request = sp.playlist(playlist_id=playlist_uri, fields="name")
        name = request['name']
        return name, df
    else:
        return df


def test_public():
    sp = setup_spotipy(
        cache_path="private/credential_spotipy",
        client_credential_fp="private/spotify_credential.json",
    )

    df = get_playlist(sp, "37i9dQZF1DX6ujZpAN0v9r")

    assert Path("private/credential_spotipy.cache").is_file()
    assert Path("private/spotify_credential.json").is_file()
    assert df.shape == (100, 65)


def test_user():
    sp = setup_spotipy(
        cache_path="private/credential_spotipy",
        client_credential_fp="private/spotify_credential.json",
        username='squarex',
        scope=['user-library-read']
    )

    df = query_liked_songs(sp, tracks_count=50)

    assert Path("private/credential_spotipy.cache-squarex").is_file()
    assert Path("private/spotify_credential.json").is_file()
    assert df.shape == (50, 73)

