"""
@author : ankit
"""
#!/bin/python
from movielens import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import math
import pickle
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error
import struct
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
# Store data in arrays
user = []
item = []
rating = []
rating_test = []

# Load the movie lens dataset into arrays
d = Dataset()
d.load_users("data/u.user", user)
d.load_items("data/u.item", item)
d.load_ratings("data/u.base", rating)
d.load_ratings("data/u.test", rating_test)

n_users = len(user)
n_items = len(item)

def setlimit(rate):
    if(rate<1.0):
        return 1.0
    else:
        return 5.0

movies=[]


for movie in item:
    movies.append([movie.unknown,movie.action,movie.adventure,movie.animation,movie.childrens,
        movie.comedy,movie.crime,movie.documentary,movie.drama,movie.fantasy,movie.film_noir,movie.horror,
        movie.musical,movie.mystery,movie.romance,movie.sci_fi,movie.thriller,movie.war,movie.western])

movies=np.array(movies)



#reduced_data=PCA(n_components=2).fit_transform(movies)

kmeans=KMeans(n_clusters=2).fit(movies)
#for i in range(0,len(kmeans.labels_)):
#   print(kmeans.labels_[i])

#print({i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)})
length=2

'''
centroids=kmeans.cluster_centers_

plt.plot(reduced_data[:,0],reduced_data[:,1],'k.',markersize=2)

plt.scatter(centroids[:,0],centroids[:,1],marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.show()


'''
utility = np.zeros((n_users, n_items))
for r in rating:
    utility[r.user_id-1][r.item_id-1] = r.rating

# Finds the average rating for each user and stores it in the user's object
for i in range(n_users):
    rated = np.nonzero(utility[i])
    n = len(rated[0])
    if n != 0:
        user[i].avg_r = np.mean(utility[i][rated])
    else:
        user[i].avg_r = 0.

#print utility
clustered_matrix=[]

for i in range(0,n_users):
    mean_matrix=[]
    for j in range(0,length):
        mean_matrix.append([])
        value=np.zeros(length)
    for j in range(0,n_items):
        if(utility[i][j]>0):
            mean_matrix[kmeans.labels_[j]-1].append(utility[i][j])
    for j in range(0,length):
        if(len(mean_matrix[j])==0):
            value[j]=0
        else:
             value[j]=np.mean(mean_matrix[j])
 
    clustered_matrix.append(value)

clustered_matrix=np.array(clustered_matrix)


for i in range(0,n_users):
    mean_matrix=clustered_matrix[i]
    count=0
    sum=0
    for j in range(0,len(mean_matrix)):
        if(mean_matrix[j]!=0):
            sum+=mean_matrix[j]
            count+=1
    if(count>0):
        user[i].avg_r=(sum/count)
    else:
        user[i].avg_r=0.0

# Finds the Pearson Correlation Similarity Measure between two users
def pcs(x, y):
    x_r=0
    y_r=0
    x_r_2=0
    y_r_2=0
    for i in range(0,length):
        if(clustered_matrix[x-1][i]!=0 and clustered_matrix[y-1][i]!=0):
            x_r+=clustered_matrix[x-1][i]-user[x-1].avg_r
            y_r+=clustered_matrix[y-1][i]-user[y-1].avg_r
    rate1=clustered_matrix[x-1]
    rate2=clustered_matrix[y-1]
    for i in range(0,length):
        if(clustered_matrix[x-1][i]!=0):
            x_r_2+=(clustered_matrix[x-1][i]-user[x-1].avg_r)*(clustered_matrix[x-1][i]-user[x-1].avg_r)
    for i in range(0,length):
        if(clustered_matrix[y-1][i]!=0):
            y_r_2+=(clustered_matrix[y-1][i]-user[y-1].avg_r)*(clustered_matrix[y-1][i]-user[y-1].avg_r)
    if(x_r_2*y_r_2!=0):
        return (x_r*y_r)/(math.sqrt(x_r_2)*math.sqrt(y_r_2))
    else:
        return 0


def normalization():
    normalize_rating=np.zeros((n_users,length))
    for i in range(0,n_users):
        for j in range(0,length):
            if(clustered_matrix[i][j]!=0):
                normalize_rating[i][j]=clustered_matrix[i][j]-user[i].avg_r
            else:
                normalize_rating[i][j]=9999999
    return normalize_rating

# Guesses the ratings that user with id, user_id, might give to item with id, i_id.
# We will consider the top_n similar users to do this

Pearson_matrix=np.zeros((n_users,n_users))

for i in range(0,n_users):
    for j in range(0,n_users):
        Pearson_matrix[i][j]=pcs(i+1,j+1)



#print(len(Pearson_matrix[0]))
def guess(user_id, i_id, top_n):
    rating_similarity=[]
    temp=normalization()
    temp = np.delete(temp, user_id-1, 0)
    
    for i in range(0,n_users):
        if(i!=user_id-1):
            rating_similarity.append(Pearson_matrix[user_id-1][i])
        

    rating_new= [x for (y,x) in sorted(zip(rating_similarity,temp), key=lambda pair: pair[0], reverse=True)]


    count=0
    sum=0
    for i in range(0,top_n):
        if(rating_new[i][i_id-1]!=9999999):
            sum=sum+rating_new[i][i_id-1]
            count=count+1
    cal_rating=0
    if(count==0):
        cal_rating=user[user_id-1].avg_r
    else:
        cal_rating=(sum/count)+user[user_id-1].avg_r

    if(cal_rating<1.0 or cal_rating >5.0):
        return setlimit(cal_rating)
    else:
        return cal_rating

final_ratings=np.array(clustered_matrix)

for i in range(0,n_users):
    for j in range(0,length):
        if(final_ratings[i][j]==0):
            final_ratings[i][j]=guess(i+1,j+1,150)




utility_test = np.zeros((n_users, n_items))
for r in rating_test:
    utility_test[r.user_id-1][r.item_id-1] = r.rating

#data=pd.read_csv("data/u.test",header=None,delimiter=r"\s+")

y_pred=[]

y_true=[]

#f=open('predictions7.txt','w')

for i in range(0,n_users):
    for j in range(0,n_items):
        if(utility_test[i][j]>0):
            y_pred.append(final_ratings[i][kmeans.labels_[j]-1])
            y_true.append(utility_test[i][j])
            #f.write("%d, %d, %.4f\n" % (i+1, j+1, final_ratings[i][kmeans.labels_[j]-1]))

#f.close()
print(mean_squared_error(y_pred,y_true))

