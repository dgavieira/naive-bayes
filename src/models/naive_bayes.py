import numpy as np

class NaiveBayes:
    def __init__(self, weights):
        self.weights = weights

    def fit(self):
        raise NotImplementedError
    
    def predict(self):
        raise NotImplementedError
    
    def cross_val_predict(self):
        raise NotImplementedError