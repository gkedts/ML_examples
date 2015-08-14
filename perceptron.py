import numpy as np
from matplotlib import pyplot as plt

f = open('iris_pca.txt','r')

raw_data = [line.replace('\n','') for line in f]
X = np.zeros((len(raw_data),2))

for i in range(len(raw_data)):
    X[i] = raw_data[i].split(',')
labeled = zip(X[:50],[-1]*50) + zip(X[50:],[1]*100)

"""
perceptron algorithm: initialize at vector [0,0], update for misclassified poinnts in training data, verify generalizability with validation data.

"""
training = labeled[1::2]
t_points = np.array([i[0] for i in training])
t_labels = np.array([i[1] for i in training])
validation = labeled[::2]
v_points = np.array([i[0] for i in validation])
v_labels = np.array([i[1] for i in validation])

theta = np.zeros((2,1))
theta_0 = 0
count = 0
iterations = 75
solved = False

while not solved and count < iterations:
    k_this_round = 0
    for i in range(len(training)):
        data = training[i][0].reshape(2,1)
        label = training[i][1]
        if label*(np.dot(data.T,theta) + theta_0) <= 0:
            theta += label*data
            theta_0 += label
        k_this_round += 1
    solved = (k_this_round == 0)
    count += 1

    #if count%5 == 0:    
    #    linex = []
    #    liney = []
    #    for x in np.linspace(0,10):
    #        linex.append(x)
    #        if theta[1] != 0:
    #            y = (-theta_0 - theta[0]*x)/theta[1]
    #            liney.append(y)
    #
    #    plt.scatter(t_points.T[0],t_points.T[1],marker='o',c=t_labels)
    #    plt.plot(linex,liney,'k-')
    #    plt.axis([0,10,0,1])
    #    plt.show()

linex = []
liney = []
for x in np.linspace(0,10):
    linex.append(x)
    if theta[1] != 0:
        y = (-theta_0 - theta[0]*x)/theta[1]
        liney.append(y)

plt.scatter(t_points.T[0],t_points.T[1],marker='o',c=t_labels)
plt.plot(linex,liney,'k-')
plt.axis([0,10,0,1])
plt.show()

plt.scatter(v_points.T[0],v_points.T[1],marker='o',c=v_labels)
plt.plot(linex,liney,'k-')
plt.axis([0,10,0,1])
plt.show()
