# Movie-Recommender-System

This  repository is about the movie recommender system

## Data set:

Dataset is taken from [here](https://grouplens.org/datasets/movielens/) 

1.Each movie has 19 features.

The data set contains user rated each movie and the features of the movie represented as 0's and 1's
1 represent movie is of the feature type whereas 0 represents movie is not

	
## Approach:

Using K-Means Clustering and Pearson correlation Similarity the users who has similar interests are identified and movies are 
recommended to the users.


## Result:

Algorithm acheived mean squared error of 1.24154447389

Optimising mean squared error:
Mean Squared error for n_clusters = 2 is 1.08155166095

The plot after applying PCA to data and clustering by Kmeans [here](https://user-images.githubusercontent.com/22453634/31859508-b7b310be-b72a-11e7-91a6-7fdcde97d2e3.png)
