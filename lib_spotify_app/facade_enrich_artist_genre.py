import numpy as np
from pathlib import Path
import spotipy
import pandas as pd
from typing import Dict, List

from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import yule
from scipy.cluster.hierarchy import (
    fcluster, dendrogram, linkage, cut_tree, leaders)
from sklearn.cluster import OPTICS

import matplotlib.pyplot as plt
from IPython.display import display

from .util import _enrich_by_feature

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
        self._artists = artists
        self._mlb = MultiLabelBinarizer()


    @property
    def genre(self)->pd.Series:
        """
        Retrieve the genres from the initial data
        
        Returns
        -------
        pd.Series
            Series of all the genre
        """
        return self.df_genre.columns


    def enrich_artists(self, sp:spotipy.Spotify)->pd.DataFrame:
        df = _enrich_by_feature(self._artists,
                               w=50,
                               f=sp.artists)

        # DataFrame with genre as a column and each row is an artist
        genre_col = f'{self._artists.name}.enrich.genres'
        self.df_genre = pd.DataFrame(
            self._mlb.fit_transform(df[genre_col]),
            columns=self._mlb.classes_,
            index=self._artists.index
        )
        
        return df

    def clean_useless_genre(self):
        """
        spotify contains strange genre such as "alabama_indie" which are not useful for our purpose. As such this method get rids of all of them
        """
        mask = self.df_genre.columns.str.contains(r'genre_\w+ indie')
        self.df_genre = self.df_genre.loc[:,~mask]


    def cluster_genre(self, method:str='average'):
        self.linkage = linkage(self.df_genre,
                               method=method,
                               metric='yule')
        
    def plot_dendrogram(self):
        dendrogram(
            self.linkage,
            orientation='left',
            labels=self.genre.to_list()
        )

    def create_super_genre(self):
        import re
        from itertools import chain
        from collections import Counter
        from statistics import mode
        pass