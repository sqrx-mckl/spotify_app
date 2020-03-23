from pathlib import Path
import spotipy
import json
from typing import Dict, List

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

        # from "_get_token"
        self.token_code = None
        # from "open_session"
        self.sp:spotipy.Spotify = None


    def _get_token(self):
        NotImplementedError('''deprecated method
        refer to https://github.com/plamere/spotipy/issues/263''')

        # token need to be refreshed
        try:
            self.token_code = spotipy.util.prompt_for_user_token(
                username=self.credential["username"],
                client_id=self.credential["client_id"],
                client_secret=self.credential["client_secret"],
                redirect_uri="http://localhost/",
                scope=self.scope,
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
        """
        Open a session with OAuth
        refer to <https://github.com/plamere/spotipy/issues/263>
        """

        ## Depreciated
        # self._get_token()
        # self.sp = spotipy.Spotify(auth=self.credential['token']['code'])

        # new method as per <https://github.com/plamere/spotipy/issues/263>

        self.sp=spotipy.Spotify(
            auth_manager=spotipy.SpotifyOAuth(
                client_id=self.credential["client_id"],
                client_secret=self.credential["client_secret"],
                redirect_uri="http://localhost/",
                scope=self.scope,
                cache_path=self.cache_path)
        )

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
            if tracks_count > 0 and offset >= tracks_count:
                break

        return result_liked_songs

