import numpy as np
from pathlib import Path
import spotipy
import pandas as pd
from typing import Dict, List, Union, Tuple
import seaborn as sns

from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import hamming
from scipy.cluster.hierarchy import (
    fcluster, dendrogram, linkage, cut_tree, leaders)
from sklearn.cluster import OPTICS
from sklearn.neighbors import LocalOutlierFactor

from hdbscan import HDBSCAN
import spacy
import re

import matplotlib.pyplot as plt
from IPython.display import display

from chord import Chord

from .util import concatenate_col
from .api_adapter import _enrich_by_feature

__all__ = [
    "add_genres",
    "EnrichArtistGenre",
    "join_genre"
]

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

    The metric used for the genre combination is the Hamming distance:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html
    It is a binary array disambleance distance. It uses the amount of time a True is encountered at the same index for both arrays

    Parameters
    ----------
    artists_id: pd.Series[List[str]]
        Spotify 
    feature : pd.DataFrame
        feature from artist request on Spotify in a DataFrame
    df_genre : pd.DataFrane
        encoded genre with as rows the artists, and columns the genre from those artists
    _artist : pd.Series
        artist data to be transformed
    _mlb : MultiLabelBinarizer
        used to encode the artist genre        
    """
    
    # get, set -----------------------------------------------------------------

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

    @property
    def supergenre_per_artists(self):
        return self.df['supergenres'].apply(lambda x: len(x))
    
    @property
    def supergenre_occurences(self):
        return self.df_supergenre.sum()

    @property
    def supergenre_count_genre(self):
        return self.supergenre_group.agg(lambda x: sum(x.notna()))


    # constructor --------------------------------------------------------------

    def __init__(
        self,
        artists_id:Union[pd.Series, pd.DataFrame],
        sp:Union[spotipy.Spotify, None]=None,
        genre:Union[pd.Series, None]=None
    ):

        if isinstance(artists_id, pd.Series):
            self._artist = artists_id
        elif isinstance(artists_id, pd.DataFrame):
            self._artist = pd.concat(
                [artists_id[col] for col in artists_id]
            ).dropna()
        self._artist = pd.Series(self._artist.unique(), name='artists_id')

        self.cluster = None
        self.df = None
        self.df_genre = None
        self.df_supergenre = None
        self.feature = None
        self.supergenre_group = None

        if sp is not None:
            # if genre unavailable, enrich with the genre and then clean it
            self.enrich_artist_features(sp)
        
        elif genre is not None:
            self.df:pd.DataFrame = pd.concat([genre, artists_id], axis=1)\
                .set_axis(['genres', 'artists'], axis=1)\
                .drop_duplicates('artists')\
                .set_index('artists')
            self._set_df_genre()
        
        # self.clean_geo_genre()
        # self.clean_genre_low_encounter()


    # methods ------------------------------------------------------------------

    def enrich_artist_features(self, sp:spotipy.Spotify):

        self.feature = _enrich_by_feature(self._artist,
                                          w=50,
                                          f=sp.artists)
        
        self.df = self.feature[['genres']]

        self._set_df_genre()


    def clean_geo_genre(self)->pd.Series:
        """
        spotify contains strange genre such as "alabama_indie" which are not useful for our purpose. As such this method get rids of all of them

        Returns
        -------
        Series of the deleted genres
        """

        nlp = spacy.load("en_core_web_sm")

        # detect in each word in the genre name if it recognizes as a country, 
        # state, language, etc... Genre should be independant of country
        def clean_geo(x:str):
            is_geo = False
            for word in x.split(' '):
                for ent in list(nlp(word).ents):
                    is_geo = is_geo or (ent.label_ in ['NORP', 'GPE', 'LOC'])
            return is_geo

        mask = self.genre.to_series().apply(clean_geo)
        self._genre_geo = self.df_genre.loc[:,mask].columns
        self.df_genre:pd.DataFrame = self.df_genre.loc[:,~mask]

        return self._genre_geo


    def clean_genre_low_encounter(self, verbose=False):
        """
        The genres without interest will usually encounter very few other 
        genres, as such we need to know how many different genres is 
        encountered by each genre. This is the following plot
        
        Parameters
        ----------
        verbose : bool, optional
            by default False
        """
        
        encounter_genre = self.df_genre.corr(lambda x,y: max(x*y))
        
        # delete the genre with too low encounters
        mask = encounter_genre.sum() >= 2

        self._genre_low_encounter = self.genre[~mask]
        self.df_genre:pd.DataFrame = self.df_genre.loc[:,mask]

        if verbose:
            display(f'There is {sum(~mask)} genres filetered')
            with pd.option_context('display.max_rows', None):
                display(pd.DataFrame(
                    self.df_genre.sum()[~mask].sort_values(ascending=False)).T)

            plt.figure()
            encounter_genre.sum().loc[mask].plot(kind='box')
            
            plt.figure()
            encounter_genre.sum().apply(np.log10).loc[mask].plot(kind='kde')


    # helper methods -----------------------------------------------------------

    def _set_df_genre(self):
        # DataFrame with genre as a column and each row is an artist
        self.df_genre, self._mlb = EnrichArtistGenre.get_df_genre(
            index=self._artist,
            genre_serie=self.df.iloc[:,0]
        )
        self._df_genre_raw = self.df_genre.copy()


    def _get_group_list(
        self,
        cluster:List[int],
        df_genre:pd.DataFrame=None
    )->pd.DataFrame:
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


    def _setup_supergenre(self):
        # Get a "semi-table" with the supergenre group
        self.supergenre_group = self._get_group_list(self.cluster)

        # name the super-genre
        supergenres = self.supergenre_group\
            .applymap(self._split_word)\
            .agg(lambda x: self._robust_str_mode(x.to_list()))\
            .tolist()
        self.supergenre_group.columns = [f'{i}_{x}' for i, x in enumerate(supergenres)]

        # merge the super-genre into the list of songs artists
        self.df_supergenre:pd.DataFrame = self.df_genre\
            .groupby(self.cluster, axis=1)\
            .max()\
            .set_axis(self.supergenre, axis=1)

        # add cluster to the table
        s_supergenres = self.df_supergenre.apply(
            lambda row: row.index[row==1].to_list(),
            axis=1
        )
        self.df = self.df.assign(supergenres=s_supergenres)

    @staticmethod
    def get_df_genre(index:pd.Index, genre_serie:pd.Series):
        mlb = MultiLabelBinarizer()
        df_genre = pd.DataFrame(
            mlb.fit_transform(genre_serie),
            columns = mlb.classes_,
            index = index
        )
        return df_genre, mlb

    @staticmethod
    def _split_word(x:str)->List[str]:
        """
        split a string as a list of words
        
        Arguments:
            x {str} -- string to split
        
        Returns:
            List[str] -- list of words (as strngs)
        """
        if isinstance(x, float) and np.isnan(x):
            return []
        return re.findall(r"[\w']+", x)


    @staticmethod
    def _robust_str_mode(x:List[List[str]], sep:str=' ')->str:
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

        word = list(chain(*x))

        # like mode() but more robust (give all modes concatenated in 1 string)
        word_count = pd.Series(word).value_counts()
        count_threshold = min([len(word_count.unique())-1, 1])
        word_count_select = word_count >= word_count.unique()[count_threshold]
        word_to_join = word_count.index[word_count_select].values
        return sep.join(word_to_join)


    # plot and analysis --------------------------------------------------------

    def _analyse_supergenre(self):
        print(f'there is {len(self.supergenre)} supergenres')

        plt.figure()
        self.df['supergenres']\
            .apply(lambda x: len(x))\
            .hist()\
            .set_title('# of supergenre per artists')

        plt.figure()
        self.supergenre_group\
            .agg(lambda x: sum(x.notna()))\
            .sort_values()\
            .plot(kind='barh')\
            .set_title('# of genre')

        plt.figure()
        self.df_supergenre\
            .sum()\
            .sort_values()\
            .plot(kind='barh')\
            .set_title('# of occurences')

        display(self.supergenre_group)

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

    @staticmethod
    def _plot_chord(df, name):
        encounter_matrix = df.corr(lambda x, y: sum(x*y)).values
            
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
            raise TypeError(f'{selection} is neither "int" nor "str"')
        
        EnrichArtistGenre._plot_chord(self.df_genre[genre_select], genre_select)

    # clustering methods -----------------------------------------------------

    def cluster_genre_fit(self, method='weighted', algorithm:str='hdscan'):
        """
        This function fit the clustering models (the hierarchy one).
        A "cluster_genre_transform()" exist to apply your specific proposal

        Clustering is done by 2 algorithms:
        * DBSCAN with OPTICS to detect the outliers
        * hierarchical clustering to create the "super-genre"

        The metric used for the genre combination is the Hamming distance:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html
        It is a binary array disambleance distance. It uses the amount of time a True is encountered at the same index for both arrays

        Parameters
        ----------
        method : str
            method for Hierarchical Clustering, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
            by default 'weighted'
        algorithm : str
            algorithm used for first clustering, DBscan. Can either be dbscan 
            which will apply OPTICS + DBscan as per sklearn method. Or HDBSCAN 
            which will apply new algorithm.
            By default hdbscan
        """

        df_genre = self.df_genre.transpose()

        if algorithm == 'dbscan':
            self._optics = OPTICS(
                max_eps=1,
                min_samples=2,# to avoid lonely genre
                metric='hamming',
                cluster_method='dbscan',
                algorithm='brute',# only valid algorithm for this metric
                n_jobs=-1
            )
            self.cluster_dbscan = self._optics.fit_predict(df_genre)
        elif algorithm == 'hdbscan':
            self._mdl_hdbscan = HDBSCAN(
                min_cluster_size=3,
                metric='hamming'
            )
            self.cluster_dbscan = self._mdl_hdbscan.fit_predict(df_genre)
        else:
            raise AttributeError(f'{algorithm} is not available, use "dbscan" or "hdbscan"')


        # remove the outlier cluster
        self._df_genre_inlier = self.df_genre.loc[:,self.cluster_dbscan > -1]
        self._df_genre_outlier = self.df_genre.loc[:,self.cluster_dbscan == -1]

        self._linkage = linkage(self._df_genre_inlier.transpose(),
                                method=method,
                                metric='hamming')
        

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
        self.cluster = self.cluster_dbscan.copy()
        self.cluster[self.cluster_dbscan > -1] = self.cluster_hierarchical

        self._setup_supergenre()

        if verbose:
            self._analyse_supergenre()


    def cluster_genre_transform_optimization(self, verbose=False):
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
                * np.log10(self.supergenre_count_genre.var())

        res = minimize_scalar(minimize_genre_per_artists,
                              options={'maxiter':10e3})

        self.cluster_genre_transform(t=res.x, criterion='distance')

        if verbose:
            print(f'distance optimal is {res.x}')
            self._analyse_supergenre()


    def cluster_genre_transform_dbscan(self, lof:bool=True, verbose:bool=False):
        """
        Cluster by DBscan directly
        
        Parameters
        ----------
        lof : bool, optional
            to apply Local Outlier Factor on top of HDBscan to clean some genres
        verbose : int, optional
            verbose analysis, by default False
        """
        self.cluster = self.cluster_dbscan.copy()
        self._setup_supergenre()

        if lof:
            # delete the genre with too low encounters
            mask = LocalOutlierFactor(n_neighbors=10, metric='hamming')\
                            .fit_predict(self.df_supergenre.T)
            mask_select = mask == -1
            select_cluster = [k for k, v in enumerate(mask_select) if ~v]

            if verbose:
                display(f'There is {sum(~mask_select)} genres filetered')

                for mask_select_plot in [~mask_select, mask_select]:
                    with pd.option_context('display.max_rows', None):
                        display(pd.DataFrame(
                            self.df_supergenre.sum()[mask_select_plot]\
                                .sort_values(ascending=False)
                        ))
            
            self.cluster[[x in select_cluster for x in self.cluster]] = -2

        self._setup_supergenre()

        if verbose:
            self._analyse_supergenre()



    def cluster_genre(self,
                      method='weighted',
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

        The metric used for the genre combination is the Hamming distance:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html
        It is a binary array disambleance distance. It uses the amount of time a True is encountered at the same index for both arrays

        
        For more details on the parameters, refer to:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
        
        Parameters
        ----------
        verbose : bool
            Set to True to obtain useful plots and displayed values to help 
            analyse the clustering algorithm fitting and transformation
            By default False
        """

        self.cluster_genre_fit(method=method)
        self.cluster_genre_transform_dbscan()

        if verbose:
            self.plot_dendrogram()
            self._analyse_supergenre()
            self.plot_chord_supergenre()


def join_genre(ser_artist_id:pd.Series, df_genre:pd.DataFrame) -> pd.Series:
    """
    Join the genre table with a series of artists ID

    Parameters
    ----------
    ser_artist_id : pd.Series
        a series with a list of artists ID as values
    df_genre : pd.DataFrame
        a genre dataframe with the genre name as columns and the artists ID as index

    Returns
    -------
    pd.Series
        a new series which contains all the genre of each artists in a list (values are list)
    """

    genre_df = df_genre.apply(
        lambda row: row.index[row == True].tolist(),
        axis=1
    )

    return ser_artist_id.map(
        lambda x: sum(genre_df.loc[x].tolist(), [])
    )
    

def add_genres(
    df,
    sp,
    col_regex='artists\.\d+\.id'
) -> Tuple[pd.DataFrame, EnrichArtistGenre]:

    genre = EnrichArtistGenre(
        artists_id=df.filter(regex=col_regex),
        sp=sp
    )
    genre.clean_geo_genre()

    mdl_hdbscan = HDBSCAN(
        min_cluster_size=16,
        min_samples=16,
        metric='hamming'
    )
    cluster_hdbscan = mdl_hdbscan.fit_predict(genre.df_genre.transpose())
    genre.cluster = cluster_hdbscan
    genre._setup_supergenre()

    # concat artists
    df['artists.id'] = concatenate_col(df, col_regex)

    # add genres
    genre_df = genre.df_genre.apply(
        lambda row: row.index[row == True].tolist(),
        axis=1
    )
    df['artists.genres'] = df['artists.id'].map(
        lambda x: sum(genre_df.loc[x].tolist(), [])
    )

    # add supergenres
    df['artists.supergenres'] = join_genre(
        df['artists.id'],
        genre.df_supergenre
    )

    # add first supergenres
    df['artists.supergenre_1'] = df['artists.supergenres'].map(
        lambda x: x[-1] if len(x) > 0 else np.NaN
    )

    return df, genre

