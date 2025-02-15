import random
import json
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import shap 
from utils import compute_log_odds
SEED = 42

np.random.seed(SEED)
random.seed(SEED)

class RFExplainer:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename
   
    def get_feature_importances(self, label_to_explain, method='tree', model='drums'):
        with open(self.path, 'r') as file:
            data = json.load(file)

        if model == 'drums':
            y = compute_log_odds(data['scores'], label_to_explain)
            y = y.tolist()
        else:
            y = [score[label_to_explain] for score in data['scores']]

        distances = data['neighborhood']

        RFModel = RandomForestRegressor(n_estimators=100, n_jobs=16)  
        RFModel.fit(data['snrs'], y, sample_weight=distances)

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
    


        