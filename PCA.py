import numpy as np
from matplotlib import pyplot as plt

f = open('iris.txt','r')

raw_data = [line for line in f]
X = np.zeros((len(raw_data),4))

for i in range(len(raw_data)):
    sepal_len,sepal_wid,petal_len,petal_wid,label = raw_data[i].split(',')
    X[i] = [sepal_len,sepal_wid,petal_len,petal_wid]

mean_vector = np.sum(X,axis=0)/len(raw_data)

scatter_matrix = np.zeros((4,4))
for i in range(X.shape[0]):
    diff = X[i].reshape(4,1) - mean_vector
    scatter_matrix += diff.dot(diff.T)

eig_val, eig_vec = np.linalg.eig(scatter_matrix)
eig_pairs = zip(eig_val,eig_vec)
eig_pairs.sort()
eig_pairs.reverse()

W = np.hstack((eig_pairs[0][1].reshape(4,1),eig_pairs[1][1].reshape(4,1)))
Y = X.dot(W).T

plt.plot(Y[0,0:50], Y[1,0:50],'o')
plt.plot(Y[0,50:100],Y[1,50:100],'^')
plt.plot(Y[0,100:],Y[1,100:],'+')

plt.show()
