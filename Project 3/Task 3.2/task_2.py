# Subtask 2: Spectral clustering
"""
@author: Valeriia Volkovaia
"""


import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.cluster import KMeans


def task_2():
    df = pd.read_csv("data-clustering-2.csv", sep = ',',  header=None) # read data in dataframe
    df = df.transpose()


    mean_x = np.mean(df[0]) # calculate mean for x axis
    mean_y = np.mean(df[1]) # calculate mean for y axis

    df[0] = (df[0]-mean_x)/np.std(df[0]) #normalizing x axis
    df[1] = (df[1]-mean_y)/np.std(df[1]) #normalizing y axis


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(df[0], df[1], 'ko')
    plt.show()

    # Kmeans from scipy
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(df)
    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_
    fig = plt.figure(figsize=(5, 5))
    colmap = {1: 'r', 2: 'g'}

    colors = map(lambda x: colmap[x+1], labels)

    plt.scatter(df[0], df[1], color=colors, alpha=0.5, edgecolor='k')
    for idx, centroid in enumerate(centroids):
        plt.scatter(*centroid, color=colmap[idx+1])
    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.show()


    #Spectral clustering
    beta = 1

    S = np.zeros((len(df[0]),len(df[1])))

    #calculate similarity matrix
    for i in xrange(len(df[0])):
        for j in xrange(len(df[1])):
            S[i][j] = math.exp(-beta*math.pow(np.linalg.norm(df.iloc[i]-df.iloc[j]),2))

    D = np.zeros((len(df[0]),len(df[1])))

    sums = S.sum(axis=1)
    np.fill_diagonal(D, sums)

    L = D - S

    #calculate eigenvalues and eigenvectors
    w, v = np.linalg.eig(L)

    #sort eigenvalues and eigenvectors
    sorted_w, sorted_v = zip(*sorted(zip(w, v)))

    fig = plt.figure(figsize=(5, 5))

    #choosing label and ploting
    i=0
    for vx in sorted_v[1]:
        if vx > 0:
            plt.scatter(df[0][i], df[1][i], color='r', alpha=0.5, edgecolor='k')
        else:
            plt.scatter(df[0][i], df[1][i], color='g', alpha=0.5, edgecolor='k')
        i += 1

    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.show()

if __name__ == "__main__":
    task_2()

