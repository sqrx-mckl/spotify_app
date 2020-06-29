from pathlib import Path
import spotipy
import json
from typing import Dict, List
import pylast
import pandas as pd

def setup_spotipy(
    credential_fp:Path,
    scope:List[str],
    cache_path:Path
) -> spotipy.Spotify:
    """
    Initialize an instance of the spotipy.Spotify API without need to know how to setup the object.

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
    cache_path : Path
        Path to the folder where to save the cache file

    Returns
    -------
    spotipy.Spotify
        Spotipy API Facade instance
    """

    scope = ' '.join(scope)
    
    with open(credential_fp) as file:
        credential = json.load(file)
    
    # save user token in ".cache-<username>" file at user defined location
    cache_path = Path(cache_path, f'.cache-{credential["username"]}')

    # new method as per <https://github.com/plamere/spotipy/issues/263>
    # open an authentification by OAuth
    return spotipy.Spotify(
        auth_manager=spotipy.SpotifyOAuth(
            client_id=credential["client_id"],
            client_secret=credential["client_secret"],
            redirect_uri="http://localhost/",
            scope=scope,
            cache_path=cache_path
        )
    )


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
        api.user:pylast.User = api.get_user(credential['username'])

        return api


def query_liked_songs(
    sp:spotipy.Spotify,
    tracks_count:int=None,
    limit:int=50
) -> pd.DataFrame:
    """
    Query the current user liked songs in a loop. Spotify API limit the number of queried songs to 50 at a time.
    
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

    result = {'items':[]}

    while True:
        # cache the previous result
        result_item_temp = result['items']

        # new query with recursive arguments
        result = sp.current_user_saved_tracks(
            limit=limit,
            offset=len(result_item_temp)
        )

        # update info using query result
        if tracks_count is None:
            tracks_count = result['total']
        
        # append the dictionnary output
        result['items'] += result_item_temp

        # update argument for last query adjustement
        limit = min([limit, tracks_count-len(result['items'])])

        if limit <= 0:
            break

    return normalize_request(result)


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
        df_list = [pd.json_normalize(_json_list2dict(r)) for r in request]
        df = pd.concat(df_list).reset_index()
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