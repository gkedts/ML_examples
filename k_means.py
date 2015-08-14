from matplotlib import pyplot as plt
import numpy as np
import random

f = open('iris_pca.txt','r')

raw_data = [line.replace('\n','') for line in f]
X = np.zeros((len(raw_data),2))

for i in range(len(raw_data)):
    X[i] = raw_data[i].split(',')

"""
k-means algorithm: randomly initialize centroids, assign points to centroids,
find means for each cluster of points, repeat.

"""
k = 3
centroids = np.zeros((k,2))
cluster_assignments = [0]*len(X)
prev_cost = 1
cost = 0

for i in range(k):
    centroids[i] = np.array([random.randint(5,10),random.random()])

while abs(prev_cost - cost) > 1e-7:
    prev_cost = cost
    cost = 0
    for i in range(len(X)):
        x = X[i]
        prev = 'none'
        for j in range(k):
            point = centroids[j]
            distance = (x[0] - point[0])**2 + (x[1] - point[1])**2
            if distance < prev or prev == 'none':
                cluster_assignments[i] = j
                prev = distance
            cost += prev

    centroids = np.zeros_like(centroids,float)
    for i in range(k):
        num_points = 0
        for j in range(len(X)):
            x = X[j]
            if cluster_assignments[j] == i:
                centroids[i] += x
                num_points += 1.0

        centroids[i] = np.divide(centroids[i],num_points)
    
    plt.scatter(X.T[0],X.T[1],marker='o',
                        c=cluster_assignments,cmap=plt.cm.coolwarm)
    plt.scatter(centroids.T[0],centroids.T[1],marker='^',s=80,
                        c=range(k),cmap=plt.cm.coolwarm)
    plt.show()

