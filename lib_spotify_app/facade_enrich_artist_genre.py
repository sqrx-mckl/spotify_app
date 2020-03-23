import numpy as np
from pathlib import Path
import spotipy
import pandas as pd
from typing import Dict, List
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import yule
from scipy.cluster.hierarchy import (
    fcluster, dendrogram, linkage, cut_tree, leaders)
from sklearn.cluster import OPTICS

import matplotlib.pyplot as plt
from IPython.display import display

from .util import _enrich_by_feature

class facade_enrich_artist_genre:
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
    method : str
        method for Hierarchical Clustering, see scipy.cluster.hierarchy.linkage
        by default 'weighted'
    feature : pd.DataFrame
        feature from artist request on Spotify in a DataFrame
    df_genre : pd.DataFrane
        encoded genre with as rows the artists, and columns the genre from those artists
    _artist : pd.Series
        artist data to be transformed
    _mlb : MultiLabelBinarizer
        used to encode the artist genre        
    """
    
    
    def __init__(self, artists:pd.Series):
        self._artist:pd.Series = artists
        self._mlb:MultiLabelBinarizer = MultiLabelBinarizer()
        self.method:str = 'weighted'


    @property
    def genre(self)->pd.Index:
        """
        Retrieve the genres from the initial data
        
        Returns
        -------
        pd.Series
            Series of all the genre
        """
        return self.df_genre.columns


    def request_artist_features(self, sp:spotipy.Spotify):
        self.feature:pd.DataFrame = _enrich_by_feature(self._artist,
                                                       w=50,
                                                       f=sp.artists)
        
        # DataFrame with genre as a column and each row is an artist
        self.df_genre = pd.DataFrame(
            self._mlb.fit_transform(self.feature['genres']),
            columns=self._mlb.classes_,
            index=self._artist.index
        )


    def clean_useless_genre(self):
        """
        spotify contains strange genre such as "alabama_indie" which are not useful for our purpose. As such this method get rids of all of them
        """
        mask = self.df_genre.columns.str.contains(r'genre_\w+ indie')
        self.df_genre:pd.DataFrame = self.df_genre.loc[:,~mask]


    def cluster_genre(self):
        """
        Clustering is done by 2 algorithms:
        * DBSCAN with OPTICS to detect the outliers
        * hierarchical clustering to create the "super-genre"

        The metric used for the genre combination is the Yule distance:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.yule.html
        It is a binary array disambleance distance. It uses the amount of time a True is encountered at the same index for both arrays
        """

        from sklearn.cluster import OPTICS

        df_genre = self.df_genre.transpose()

        self.cl_optics = OPTICS(
            max_eps=1,
            min_samples=2,# to avoid lonely genre, goal is to have super-genre
            metric='yule',
            cluster_method='dbscan',
            algorithm='brute',# only valid algorithm for this metric
            n_jobs=-2
        )
        self.cl_optics = self.cl_optics.fit_predict(df_genre)

        # remove the outlier cluster
        self.cl_linkage = linkage(df_genre,
                            method=self.method,
                            metric='yule')
        

    def plot_clustermap(self):
        plt.figure()
        g = sns.clustermap(df_genre,
                           row_linkage=self.cl_linkage,
                           col_cluster=False,
                           yticklabels=True)
        g.fig.suptitle(self.method)


    def plot_dendrogram(self):
        dendrogram(
            self.cl_linkage,
            orientation='left',
            labels=self.genre.to_list()
        )


    def create_super_genre(self):
        from itertools import chain

        
        pass


    def enrich_artist_genre(self, df:pd.DataFrame)->pd.DataFrame:
        pass


@staticmethod
def _split_word(x:str)->List[str]:
    """
    split a string as a list of words
    
    Arguments:
        x {str} -- string to split
    
    Returns:
        List[str] -- list of words (as strngs)
    """
    import re

    if x is np.nan:
        return []
    x = x.replace('genre_', '')
    return re.findall(r"[\w']+", x)


@staticmethod
def _robust_str_mode(x:List[str], sep:str='-')->str:
    """
    robust method to calculate the mode of a list of string, returns a concatenate string in case of equal number of counter
    
    Arguments:
        x {List[str]} -- list of string to calcualte the mode
        sep {str} -- string used to concatenate mutliple string in case of draw
    
    Returns:
        [str] -- mode word
    """
    from statistics import mode
    
    try:
        return mode(x)
    except:
        vc = pd.Series(x).value_counts()
        x_to_join = vc.index[vc == vc.max()].values
        return '-'.join(x_to_join)

