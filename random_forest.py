import random
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import numpy as np
import os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class RandomForest:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename


    def train_model(self, label_to_explain):
        with open(self.path, 'r') as file:
            data = json.load(file)

        y = []
        for score in data['scores']:
            y.append(score[label_to_explain])

        RFModel = RandomForestRegressor(n_estimators=10, n_jobs=16)  
        RFModel.fit(data['snrs'], y, sample_weight=data['neighborhood'])

        joblib.dump(RFModel, f'/home/cbolanos/experiments/audioset/{self.filename}/rf_100_model_{label_to_explain}.pkl')

    def get_feature_importances(self, label_to_explain):
        self.train_model(label_to_explain)

        RFModel_loaded = joblib.load(f'/home/cbolanos/experiments/audioset/{self.filename}/rf_100_model_{label_to_explain}.pkl')
        importances = RFModel_loaded.feature_importances_
        
        return importances
        
    


