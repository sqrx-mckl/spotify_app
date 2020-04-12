import numpy as np
from pathlib import Path
import spotipy
import pandas as pd
from typing import Dict, List, Union
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import yule
from scipy.cluster.hierarchy import (
    fcluster, dendrogram, linkage, cut_tree, leaders)
from sklearn.cluster import OPTICS

import hdbscan
import spacy
import re

import matplotlib.pyplot as plt
from IPython.display import display

from chord import Chord

from .util import _enrich_by_feature

class EnrichArtistGenre:
    """
    Facade class which handle the artists data and most particurarly the genre information from the artist.
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
    feature : pd.DataFrame
        feature from artist request on Spotify in a DataFrame
    df_genre : pd.DataFrane
        encoded genre with as rows the artists, and columns the genre from those artists
    _artist : pd.Series
        artist data to be transformed
    _mlb : MultiLabelBinarizer
        used to encode the artist genre        
    """
    
    
    def __init__(self,
                 artists:pd.Series,
                 sp:Union[spotipy.Spotify, None]=None,
                 genre:Union[pd.Series, None]=None):
        self._artist:pd.Series = pd.Series(artists.unique(),
                                           name = artists.name)

        self.cluster = None
        self.df = None
        self.df_genre = None
        self.df_supergenre = None
        self.feature = None
        self.supergenre_group = None

        if sp is not None:
            # if genre unavailable, enrich with the genre and then clean it
            self.enrich_artist_features(sp)
            self.clean_useless_genre()
        elif genre is not None:
            self.df:pd.DataFrame = pd.concat([genre, artists], axis=1)\
                .set_axis(['genre', 'artists'], axis=1)\
                .drop_duplicates('artists')\
                .drop('artists', axis=1)
            self._set_df_genre()

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


    @property
    def supergenre(self):
        return self.supergenre_group.columns

    @staticmethod
    def get_df_genre(index:pd.Index, genre_serie:pd.Series):
        mlb = MultiLabelBinarizer()
        df_genre = pd.DataFrame(
            mlb.fit_transform(genre_serie),
            columns = mlb.classes_,
            index = index
        )
        return df_genre, mlb

    def enrich_artist_features(self, sp:spotipy.Spotify):
        
        self.feature:pd.DataFrame = _enrich_by_feature(self._artist,
                                                       w=50,
                                                       f=sp.artists)
        
        self.df:pd.DataFrame = self.feature[['genres']]

        self._set_df_genre()

    def _set_df_genre(self):
        # DataFrame with genre as a column and each row is an artist
        self.df_genre, self._mlb = EnrichArtistGenre.get_df_genre(
            index=self._artist.index,
            genre_serie=self.df.iloc[:,0]
        )
        self._df_genre_raw = self.df_genre.copy()


    def clean_useless_genre(self)->pd.Series:
        """
        spotify contains strange genre such as "alabama_indie" which are not useful for our purpose. As such this method get rids of all of them

        Returns
        -------
        Series of the deleted genres
        """

        nlp = spacy.load("en_core_web_sm")

        # detect in each word in the genre name if it recognizes as a country, 
        # state, language, etc... Genre should be independant of country
        def clean_geo(x:pd.Series):
            for ent in nlp(x).ents:
                is_geo = ent.label_ in ['NORP', 'GPE']
            return any(is_geo)

        mask = self.genre.to_series().apply(clean_geo)
        self._genre_cleaned = self.df_genre.loc[:,mask].columns
        self.df_genre:pd.DataFrame = self.df_genre.loc[:,~mask]

        return self._genre_cleaned


    def cluster_genre_fit(self, method='weighted', algorithm:str='dscan'):
        """
        This function fit the clustering models (the hierarchy one).
        A "cluster_genre_transform()" exist to apply your specific proposal

        Clustering is done by 2 algorithms:
        * DBSCAN with OPTICS to detect the outliers
        * hierarchical clustering to create the "super-genre"

        Strangely, DBscan seems to perform better in this metric...

        The metric used for the genre combination is the Yule distance:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.yule.html
        It is a binary array disambleance distance. It uses the amount of time a True is encountered at the same index for both arrays

        Parameters
        ----------
        method : str
            method for Hierarchical Clustering, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            by default 'weighted'
        """

        df_genre = self.df_genre.transpose()

        if algorithm is 'dbscan':
            self._optics = OPTICS(
                max_eps=1,
                min_samples=2,# to avoid lonely genre
                metric='yule',
                cluster_method='dbscan',
                algorithm='brute',# only valid algorithm for this metric
                n_jobs=-1
            )
            self.cluster_dbscan = self._optics.fit_predict(df_genre)
        elif algorithm is 'hdbscan':
            self._mdl_hdbscan = hdbscan.HDBSCAN(
                min_cluster_size=2,
                metric='yule'
            )
            self.cluster_dbscan = self._mdl_hdbscan.fit_predict(df_genre)
        else:
            raise AttributeError(f'{algorithm} is not available, use "dbscan" or "hdbscan"')


        # remove the outlier cluster
        self._df_genre_inlier = self.df_genre.loc[:,self.cluster_dbscan > -1]
        self._df_genre_outlier = self.df_genre.loc[:,self.cluster_dbscan == -1]

        self._linkage = linkage(self._df_genre_inlier.transpose(),
                                method=method,
                                metric='yule')
        

    def _get_group_list(self,
                        cluster:List[int],
                        df_genre:pd.DataFrame=None)->pd.DataFrame:
        """
        Convert a list of cluster and a DataFrame into a "semi-table" which 
        contains in each column all the genre for 1 supergenre. The size of 
        each column is different, so filled with NaN at the end.
        
        Parameters
        ----------
        cluster : List[int]
            List of group as int
        df_genre : pd.DataFrame
            by default, self.df_genre
            Can be another dataframe which complies to the format such as
            self.df_supergenre
        
        Returns
        -------
        pd.DataFrame
            "semi-table" with in each column the list of genre in a supergenre
        """

        if df_genre is None:
            df_genre = self.df_genre

        groups = df_genre.columns\
                         .to_frame()\
                         .reset_index(drop=True)\
                         .assign(cluster=cluster)\
                         .pivot(columns='cluster')

        # reduce row size, only need a dropping list of genres
        groups = groups.apply(
            lambda x: x.sort_values().reset_index(drop=True)
        )
        groups = groups.dropna(how='all')

        return groups


    def cluster_genre_transform(
        self,t:float=20, criterion='maxclust', verbose=False, **kwargs):
        """
        Cut the linkage matrix of the hierarchical cluster using criterion.
        For better understanding, use the available plotting 
        functions such as:
        * plot_clustermap()
        * plot_dendrogram()

        For more details on the parameters, refer to:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
        
        Parameters
        ----------
        t : float
            this is the threshold to apply when forming flat clusters.
            By default 20
        criterion : str
            "scipy.cluster.hierarchy.fcluster" argument
            By default "maxclust"

        """
        
        # Flat cluster from the hierarchy cluster algorithm
        self.cluster_hierarchical = fcluster(self._linkage,
                                             t,
                                             criterion=criterion,
                                             **kwargs)

        # Add the outliers selected by DBSCAN
        self.cluster = self.cluster_dbscan
        self.cluster[self.cluster > -1] = self.cluster_hierarchical

        # Get a "semi-table" with the supergenre group
        self.supergenre_group = self._get_group_list(self.cluster)

        # name the super-genre
        self.supergenre_group.columns = self.supergenre_group\
            .applymap(self._split_word)\
            .agg(lambda x: self._robust_str_mode(x.to_list()))
        self.supergenre_group.columns = \
            ['outliers'] + self.supergenre_group.columns.to_list()[1:]

        # merge the super-genre into the list of songs artists
        self.df_supergenre:pd.DataFrame = self.df_genre.groupby(self.cluster,   
                                                                axis=1)\
            .max()\
            .set_axis(self.supergenre, axis=1)

        # add cluster to the table
        self.df['supergenres'] = self.df_supergenre.apply(
            lambda row: row.index[row==1].to_list(),
            axis=1
        ).to_list()

        if verbose:
            print(f'there is {len(self.supergenre)} supergenres')
            self._analyse_supergenre()


    def cluster_genre_transform_auto(self, verbose=False):
        """
        This function fit the clustering models (the hierarchy one).
        Cut the linkage matrix of the hierarchical cluster using 'maxclust' 
        criterion. For better understanding, use verbose argument.
        This algorithm runs an optimization to achieve the 
        "supergenre_per_artists" criteria while keeping the number of clusters at a maximum

        For more details on the method, refer to:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
        
        Parameters
        ----------
        verbose : bool
            Set to True to obtain useful plots and displayed values to help 
            analyse the clustering algorithm fitting and transformation
            By default False
        """

        from scipy.optimize import minimize_scalar

        # maximize the number of cluster while minimizing the number of 
        # supergenre per artists

        def minimize_genre_per_artists(x):
            self.cluster_genre_transform(t=x, criterion='distance')
            return self.supergenre_per_artists.mean() \
                / np.log10(len(self.supergenre))\
                * np.log10(self.supergenre_count_genre.max())

        res = minimize_scalar(minimize_genre_per_artists,
                              options={'maxiter':10e3})

        self.cluster_genre_transform(t=res.x, criterion='distance')

        # re-apply dbscan


        if verbose:
            print(f'distance optimal is {res.x}')
            print(f'there is {len(self.supergenre)} supergenres')
            self._analyse_supergenre()

    def cluster_genre(self,
                      method='weighted',
                      supergenre_per_artists:int=2,
                      fit:bool=True,
                      verbose:bool=False,
                      **kwargs):
        """
        This function fit the clustering models (the hierarchy one).
        Cut the linkage matrix of the hierarchical cluster using 'maxclust' 
        criterion. For better understanding, use verbose argument.
        This algorithm runs an optimization to achieve the 
        "supergenre_per_artists" criteria while keeping the number of clusters at a maximum

        Clustering is done by 2 algorithms:
        * DBSCAN with OPTICS to detect the outliers
        * hierarchical clustering to create the "super-genre"

        The metric used for the genre combination is the Yule distance:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.yule.html
        It is a binary array disambleance distance. It uses the amount of time a True is encountered at the same index for both arrays

        
        For more details on the parameters, refer to:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
        
        Parameters
        ----------
        method : str
            method for Hierarchical Clustering, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            by default 'weighted'
        t : float
            this is the threshold to apply when forming flat clusters.
            By default 10
        supergenre_per_artists : int
            how many supergenre per artists do we expect to achieve by 
            optimizaton
            by default 2
        fit : bool
            To fit the clustering algorithm (linkage), by default True
        verbose : bool
            Set to True to obtain useful plots and displayed values to help 
            analyse the clustering algorithm fitting and transformation
            By default False
        """

        if fit:
            self.cluster_genre_fit(method=method)

        self.cluster_genre_transform(t=1, criterion='inconsistent')

        if verbose:
            self.plot_clustermap()
            self.plot_dendrogram()
            self._analyse_supergenre()
            self.plot_heatmap_supergenre()

    def _analyse_supergenre(self):
        plt.figure()
        self.df['supergenres']\
            .apply(lambda x: len(x))\
            .hist()\
            .set_title('# of supergenre per artists')

        plt.figure()
        self.supergenre_group\
            .agg(lambda x: sum(x.notna()))\
            .plot(kind='barh')\
            .set_title('# of genre')

        plt.figure()
        self.df_supergenre\
            .sum()\
            .plot(kind='barh')\
            .set_title('# of occurences')

        display(self.supergenre_group)

    @property
    def supergenre_per_artists(self):
        return self.df['supergenres'].apply(lambda x: len(x))
    
    @property
    def supergenre_occurences(self):
        return self.df_supergenre.sum()

    @property
    def supergenre_count_genre(self):
        return self.supergenre_group.agg(lambda x: sum(x.notna()))


    @staticmethod
    def _split_word(x:str)->List[str]:
        """
        split a string as a list of words
        
        Arguments:
            x {str} -- string to split
        
        Returns:
            List[str] -- list of words (as strngs)
        """
        if x is np.nan:
            return []
        return re.findall(r"[\w']+", x)


    @staticmethod
    def _robust_str_mode(x:List[List[str]], sep:str='_')->str:
        """
        robust method to calculate the mode of a list of string, returns a 
        concatenate string in case of equal number of counter
        
        Arguments:
            x {List[List[str]]} -- list of string to calcualte the mode
            sep {str} -- string used to concatenate mutliple string in case of 
            draw
        
        Returns:
            [str] -- mode word
        """
        from itertools import chain

        x = list(chain(*x))

        # like mode() but more robust (give all modes concatenated in 1 string)
        vc = pd.Series(x).value_counts()
        x_to_join = vc.index[vc == vc.max()].values
        return sep.join(x_to_join)


    def plot_clustermap(self):
        plt.figure()
        g = sns.clustermap(self.df_genre.transpose(),
                           row_linkage=self._linkage,
                           col_cluster=False,
                           yticklabels=True)
        # g.fig.suptitle(f'method = {self.method}')

    def plot_heatmap_supergenre(self):
        plt.figure()
        g = sns.heatmap(self.df_supergenre.transpose(),
                           yticklabels=True)


    def plot_dendrogram(self):
        plt.figure()
        dendrogram(
            self._linkage,
            orientation='left',
            labels=self.genre.to_list()
        )

    def test_supergenre_distance(self):
        self._test_supergenre('distance', list(np.arange(0, 2.05, 0.05)))

    def test_supergenre_maxclust(self):
        self._test_supergenre('maxclust', list(range(0, len(self.genre)+1)))

    def _test_supergenre(self, criterion, plt_x):
        def test_cluster_size(x:float):
            self.cluster_genre_transform(t=x, criterion=criterion)
            return [self.supergenre_count_genre.mean(),
                    self.supergenre_per_artists.mean(),
                    len(self.supergenre)]

        test_res = [test_cluster_size(x) for x in plt_x]

        plt.figure()
        plt.plot(plt_x, [x[0] for x in test_res])
        plt.legend(['supergenre_count_genre'])
        plt.figure()
        plt.plot(plt_x, [x[1] for x in test_res])
        plt.legend(['supergenre_per_artists'])
        plt.figure()
        plt.plot(plt_x, [x[2] for x in test_res])
        plt.legend(['number of supergenre'])


    def enrich_df(self, df:pd.DataFrame)->pd.DataFrame:
        prefix = self._artist.name.split('.')
        prefix = '.'.join(prefix[:-1]) + '.'

        return df.join(self.df.add_prefix(prefix),
                       on=self._artist.name)#\

    @staticmethod
    def _plot_chord(df, name):
        encounter_matrix = df.corr(lambda x, y: sum(x*y))\
                             .values
            
        encounter_matrix = encounter_matrix - np.identity(len(name))
        encounter_matrix = encounter_matrix.tolist()

        Chord(encounter_matrix, name.to_list()).show()

    def plot_chord_genre(self):
        EnrichArtistGenre._plot_chord(self.df_genre, self.genre)

    def plot_chord_supergenre(self):
        EnrichArtistGenre._plot_chord(self.df_supergenre, self.supergenre)

    def plot_chord_genre_from_supergenre(self, selection:Union[str, int]):
        if isinstance(selection, int):
            genre_select = self.df_supergenre.iloc[:, selection]
        elif isinstance(selection, str):
            genre_select = self.df_supergenre[selection]
        else:
            TypeError(f'{selection} is neither "int" nor "str"')
        
        EnrichArtistGenre._plot_chord(self.df_genre[genre_select],
                                     genre_select)
