# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 23:45:28 2017

@author: jiangjiawei
"""

import time
#import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_iris
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_svmlight_files
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_gaussian_quantiles

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from thread.train_thread import train_thread

def check_state(depth, n_classes, states):
    node_id = int((np.power(n_classes, depth) - 1) / (n_classes - 1))
    print("Check state of depth: " + str(depth) + ", start node: " + str(node_id))
    node_num = np.power(n_classes, depth)
    flag = False
    while not flag:
        finished = True
        for i in np.arange(node_num):
            #print("Check state of node: " + str(node_id+i))
            if states[node_id + i] == 0:
                finished = False
                break
        flag = finished
        time.sleep(3)
    print("Finished checking, thread state: " + str(states))

def eval_test(learners, max_depth, n_classes, X_test, y_test):
    node_ind = np.zeros(y_test.shape, dtype=int)
    y_pred = np.zeros(y_test.shape, dtype=int)
    for i in np.arange(X_test.shape[0]):
        depth = 0
        while depth < max_depth:
            cur_node = node_ind[i]
            y_pred[i] = learners[cur_node].predict(X_test[i].reshape(1, -1))[0]
            if i % 10000 == 0:
                print("current node: %d, pred: %d" % (cur_node, y_pred[i]))
            if n_classes == 2:
                node_ind[i] = n_classes * cur_node + 1 + (1 if y_pred[i] > 0 else 0)
            else:
                node_ind[i] = n_classes * cur_node + 1 + y_pred[i]
            depth = depth + 1

    #print("Accuracy of stacked LR: %s" % str(f1_score(y_test, y_pred.reshape(-1,1), average=None)))
    print("Accuracy of stacked LR: %f" % accuracy_score(y_test, y_pred))
    

t0 = time.time()

#dataset = fetch_mldata('MNIST original')
#dataset = fetch_mldata('iris')
#dataset = load_iris()

#X = dataset.data.astype('float64')
#y = dataset.target

#X, y = load_svmlight_file("/Users/jiangjiawei/dataset/w8a/w8a.train")

n_classes = 3
max_depth = 3

X, y = make_classification(n_samples=50000, n_features=200, n_informative = 180, n_classes=n_classes, flip_y= 0.2)
#X, y = make_circles(n_samples = 10000, noise=0.2, factor=0.5, random_state=1)
#X, y = make_moons(n_samples = 10000, noise=0.2, random_state=1)
#X, y = make_gaussian_quantiles(n_samples = 10000, n_features=2, n_classes=n_classes)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape(X.shape[0], -1)
y.reshape(1, -1)

print(X.shape)
print(y.shape)

n_samples = X.shape[0]
min_instance = n_samples/1000
train_samples = int(n_samples * 0.8)
test_samples = n_samples - train_samples

X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_samples, test_size=test_samples)

scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)

ind = np.zeros(y_train.shape)

max_node_num = 0
learners = []

for depth in np.arange(max_depth):
    max_node_num = max_node_num + np.power(n_classes, depth)
   
print("Max node number: " + str(max_node_num))    

for node_id in np.arange(max_node_num):
    learners.append(LogisticRegression(C=50. / n_samples,
                         multi_class='multinomial',
                         penalty='l2', solver='saga', tol=1e-4))

thread_states = np.zeros(max_node_num, dtype=int)
print("Initial thread state: " + str(thread_states))

node_id = 0

for depth in np.arange(max_depth):
    print("Depth of tree:" + str(depth))
    node_num = np.power(n_classes, depth) # start node id
    for node in np.arange(node_num):
        print("Node of tree: " + str(node_id))
        thread = train_thread(learners, X_train, y_train, ind, node_id, n_classes, thread_states)
        thread.start()
        node_id = node_id + 1
    check_state(depth, n_classes, thread_states)
    #print("Thread status: " + str(thread_states))

eval_test(learners, max_depth, n_classes, X_test, y_test)

# one LR model
base_learner = LogisticRegression(C=50. / n_samples,
                         multi_class='multinomial',
                         penalty='l2', solver='saga', tol=1e-4)
base_learner.fit(X_train, y_train)
y_pred = base_learner.predict(X_test)
print("LogisticRegression: %f" % accuracy_score(y_test, y_pred))