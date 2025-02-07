import random
import json
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import shap 

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

class RFExplainer:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
        self.kernel_width = 0.25

    def kernel(self, d):
        return np.sqrt(np.exp(-(d ** 2) / self.kernel_width ** 2))

    def get_feature_importances(self, label_to_explain, method='tree'):
        with open(self.path, 'r') as file:
            data = json.load(file)

        y = []
        for score in data['scores']:
            y.append(score[label_to_explain])

        distances = data['neighborhood']
        weights = self.kernel(distances)
       
        RFModel = RandomForestRegressor(n_estimators=100, n_jobs=16)  
        RFModel.fit(data['snrs'], y, sample_weight=weights)

        if method == 'tree':
            importances = RFModel.feature_importances_
        
        elif method == 'shap':
            with open(self.path, 'r') as file:
                data = json.load(file)
            explainer = shap.TreeExplainer(RFModel)
            first_snr = np.array(data['snrs'][0])
            choosen_instance = np.concatenate([first_snr, [data['scores'][0][label_to_explain]]]).reshape(1, -1)
    
            shap_values = explainer.shap_values(choosen_instance)
            importances = shap_values[0]

        return importances
    


        