
import sys
import os
import time

import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


class TrainLR:
    
    def __init__(self):
        self.parse_args(sys.argv)
        
    @staticmethod
    def run():
        trainer = TrainLR()
        trainer.train()
        
    def parse_args(self, argv):
        """parse args and set up instance variables"""
        try:
            self.working_dir = os.getcwd()
            self.file_name = argv[1]            
            self.batch_ratio = float(argv[2])
        except:
            print(self.usage())
            sys.exit(1)
    
    def train(self):
        start_time = time.time()
        
        classifier = SGDClassifier(loss="log")
        
        classes = np.unique(["1", "-1"])
        
        train_X, train_y = load_svmlight_file(self.file_name)
        train_X, train_y = shuffle(train_X, train_y)
        
        print("Load data cost: " + str(time.time()-start_time) + " seconds")
        
        train_start_time = time.time()
        
        minibatch_size = int(train_X.shape[0] * self.batch_ratio)
        print("Mini batch size: " + str(minibatch_size))
        
        for i in range(0, train_X.shape[0] - minibatch_size, minibatch_size):
            batch_X = train_X[i : i + minibatch_size]
            batch_y = train_y[i : i + minibatch_size]
            classifier.fit(batch_X, batch_y)
            print("Finish one batch")
        
        end_time = time.time()
        
        print("Train cost : " + str(end_time - train_start_time) + " seconds")
        
    def usage(self):
        return """
        Train a Logistic Regression with libsvm file
        Usage:
            $ python TrainLR.py <file_name> <batch_ratio>
        """
if __name__ == "__main__":
    TrainLR.run()
        




