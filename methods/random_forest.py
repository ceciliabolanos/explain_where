import random
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import shap 

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class RandomForest:
    def __init__(self, path, filename, mask_percentage, window_size):
        self.path = path
        self.filename = filename
        self.mask_percentage = mask_percentage
        self.window_size = window_size

    def train_model(self, label_to_explain):
        with open(self.path, 'r') as file:
            data = json.load(file)

        y = []
        for score in data['scores']:
            y.append(score[label_to_explain])

        RFModel = RandomForestRegressor(n_estimators=100, n_jobs=16)  
        RFModel.fit(data['snrs'], y, sample_weight=data['neighborhood'])

        joblib.dump(RFModel, f'/home/cbolanos/experiments/audioset_audios_eval/{self.filename}/rf_model_{label_to_explain}_p{self.mask_percentage}_m{self.window_size}.pkl')

    def get_feature_importances(self, label_to_explain, method='tree'):
        self.train_model(label_to_explain)

        RFModel_loaded = joblib.load(f'/home/cbolanos/experiments/audioset_audios_eval/{self.filename}/rf_model_{label_to_explain}_p{self.mask_percentage}_m{self.window_size}.pkl')

        if method == 'tree':
            importances = RFModel_loaded.feature_importances_
        
        elif method == 'shap':
            with open(self.path, 'r') as file:
                data = json.load(file)
            explainer = shap.TreeExplainer(RFModel_loaded)
            first_snr = np.array(data['snrs'][0])
            choosen_instance = np.concatenate([first_snr, [data['scores'][0][label_to_explain]]]).reshape(1, -1)
    
            shap_values = explainer.shap_values(choosen_instance)
            importances = shap_values[0]

        return importances
    

            


