import json 
import numpy as np

class GreedyExplainer:
    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
    
    def get_feature_importance(self):
        with open(self.path, 'r') as file:
            data = json.load(file)
        
        indexes = data['scores']
        N = len(indexes)
        importance_vector = np.zeros(N)

        importance_values = [1 - (i / (N - 1)) for i in range(N)]

        for idx, importance in zip(indexes, importance_values):
            idx = int(idx)
            importance_vector[idx] = importance

        return importance_vector
        

