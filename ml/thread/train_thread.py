#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:43:54 2017

@author: jiangjiawei
"""

import threading
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class train_thread(threading.Thread):
    
    def __init__(self, learners, X_train, y_train, ind, node_id, n_classes, states):
        threading.Thread.__init__(self)
        self.learners = learners
        self.learner = learners[node_id]
        self.X_train = X_train
        self.y_train = y_train
        self.ind = ind
        self.node_id = node_id
        self.n_classes = n_classes
        self.states = states
    
    def run(self):
        
        data_ind = np.where(self.ind == self.node_id)
        
        print("Start training tree node[%d], data size: %d, nodes of instance: %s"
              % (self.node_id, len(data_ind[0]), str(self.ind)))


        self.learner.fit(self.X_train[data_ind], self.y_train[data_ind])
        self.learners[self.node_id]= self.learner
        
        #sparsity = np.mean(self.learner.coef_ == 0) * 100
        #print("Sparsity with L1 penalty: %.2f%%" % sparsity)
        score = self.learner.score(self.X_train[data_ind], self.y_train[data_ind])
        print("Node[%d] test score : %.4f" % (self.node_id, score))
        
        y_pred = self.learner.predict(self.X_train[data_ind])
        #print("Labels: " + str(self.y_train[data_ind]))
        #print("Predictions: " + str(y_pred))

        child_id = self.node_id * self.n_classes + 1
        print("Childeren node starts at " + str(child_id))
        
        for i in np.arange(data_ind[0].shape[0]):
            data_id = data_ind[0][i]
            pred = y_pred[i]
            if self.n_classes == 2:
                new_node_id = child_id + (1 if pred > 0 else 0)
            else:
                new_node_id = child_id + pred
            self.ind[data_id] = new_node_id
            
        print("Node " + str(self.node_id) + ", child nodes : " + str(self.ind))
        
        self.states[self.node_id] = 1
        #print("Thread status: " + str(self.states))

            
        