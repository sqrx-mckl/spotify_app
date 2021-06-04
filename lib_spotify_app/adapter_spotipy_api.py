from pathlib import Path
import spotipy
import json
from typing import Dict, List

class SpotipyApi:
    """
    Adapter class to adapter spotipy for our usage and this specific 
    configuration
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
    sp: the Spotipy session

    """
    
    def __init__(self, credential_fp:Path, scope:List[str], cache_path:Path):
        self.scope = ' '.join(scope)
        self.credential_fp = credential_fp
        
        with open(credential_fp) as file:
            self.credential = json.load(file)
        # save user token in ".cache-<username>" file at user defined location
        self.cache_path = Path(cache_path,
                                f'.cache-{self.credential["username"]}')

        # new method as per <https://github.com/plamere/spotipy/issues/263>
        # open an authentification by OAuth
        self.sp=spotipy.Spotify(
            auth_manager=spotipy.SpotifyOAuth(
                client_id=self.credential["client_id"],
                client_secret=self.credential["client_secret"],
                redirect_uri="http://localhost/",
                scope=self.scope,
                cache_path=self.cache_path
            )
        )


    def query_liked_songs(self, tracks_count:int=None, limit:int=50):
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
        [type]
            The results as in a list of JSON request results
        """

        result = {'items':[]}

        while True:
            # cache the previous result
            result_item_temp = result['items']

            # new query with recursive arguments
            result = self.sp.current_user_saved_tracks(
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

        return result


    # //NOTE
    # Depreciated code which was very useful and with a few hacks
    # could explain why open_session() exists...
    # to delete in a few commits

    # def _get_token(self):
    #     NotImplementedError('''deprecated method
    #     refer to https://github.com/plamere/spotipy/issues/263''')

    #     # token need to be refreshed
    #     try:
    #         self.token_code = spotipy.util.prompt_for_user_token(
    #             username=self.credential["username"],
    #             client_id=self.credential["client_id"],
    #             client_secret=self.credential["client_secret"],
    #             redirect_uri="http://localhost/",
    #             scope=self.scope,
    #             cache_path=self.cache_path
    #         )
    #     except:
    #         print("error with token retrieval")
    #         raise

    #     self.credential["token"] = {'code':self.token_code,
    #                                 'scope':self.scope}
    #     with open(self.credential_fp, 'w') as file:
    #         json.dump(self.credential, file)


    # def open_session(self):
    #     """
    #     Open a session with OAuth
    #     refer to <https://github.com/plamere/spotipy/issues/263>
    #     """

    #     ## Depreciated
    #     # self._get_token()
    #     # self.sp = spotipy.Spotify(auth=self.credential['token']['code'])

    #     # new method as per <https://github.com/plamere/spotipy/issues/263>

    #     self.sp=spotipy.Spotify(
    #         auth_manager=spotipy.SpotifyOAuth(
    #             client_id=self.credential["client_id"],
    #             client_secret=self.credential["client_secret"],
    #             redirect_uri="http://localhost/",
    #             scope=self.scope,
    #             cache_path=self.cache_path
    #         )
    #     )

