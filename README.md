# spotify_app
Creation of a simple spotify app that can cluster the "liked songs" playlist from user preference.

## Final output

Dynamic plot of each top3 songs in each cluster:

[Notebook short version link](https://nbviewer.jupyter.org/github/sqrx-mckl/spotify_app/blob/master/3short_eda_umap_clustering.ipynb#My-most-popular-songs-in-a-2D-plot)

[Notebook complete version link](https://nbviewer.jupyter.org/github/sqrx-mckl/spotify_app/blob/master/3_eda_umap_clustering.ipynb#My-most-popular-songs-in-a-2D-plot)

## Example of output

Here is a static plot, all songs in a 2D projected view (with UMAP) and clustered with HDBSCAN and mahalanobis metric:

#### All songs in a 2D plot
![All songs in a 2D plot](img/umap_clusters.svg?raw=true "UMAP")
#### Audio features per Cluster
![Audio features per Cluster](img/cluster_audio_features.svg?raw=true "Audio Features Cluster")
#### Audio features in the projection
![Audio features in the projection](img/audio_features_projection.png?raw=true "Audio features in the projection")
#### Song visualization with playback
![Song visualization with playback](img/song_visualization.png?raw=true "Song visualization with playback")