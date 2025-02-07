import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
import json
from math import comb

def pi_x_for_list(vectors):
    results = []
    for x in vectors:
        m = len(x)
        z = sum(x) 
        if z == 0 or z == m:
            results.append(0) 
            continue

        binom_mz = comb(m, z)
        result = (m - 1) / (binom_mz * z * (m - z))
        results.append(result)
    
    return results

class KernelBase:
    def __init__(self, verbose=False, absolute_feature_sort=False, random_state=None):
        self.verbose = verbose
        self.absolute_feature_sort = absolute_feature_sort
        self.random_state = check_random_state(random_state)

    def explain_instance_with_data(self, neighborhood_data, neighborhood_labels, 
                                 distances, model_regressor=None):
        
        weights = pi_x_for_list(distances)
        
        if model_regressor is None:
            model_regressor = LinearRegression(fit_intercept=True)
        
        features = range(neighborhood_data.shape[1])
        model_regressor.fit(neighborhood_data[:, features],
                          neighborhood_labels,
                          sample_weight=weights)
        
        prediction_score = model_regressor.score(
            neighborhood_data[:, features],
            neighborhood_labels,
            sample_weight=weights
        )
        local_pred = model_regressor.predict(neighborhood_data[0, features].reshape(1, -1))
        
        
        return (model_regressor.intercept_,
                model_regressor.coef_,
                prediction_score,
                local_pred)

class Explanation:
    """Container for audio explanation results."""
    
    def __init__(self, neighborhood_data, neighborhood_labels):
        self.neighborhood_data = neighborhood_data
        self.neighborhood_labels = neighborhood_labels
        self.intercept = 0
        self.local_exp = {}
        self.score = None
        self.local_pred = None

    def get_feature_importances(self):
        exp = self.local_exp
    
        return {
            "coefficients": exp.tolist() if hasattr(exp[0], 'tolist') else exp,
            "local_pred": self.local_pred.tolist() if hasattr(self.local_pred, 'tolist') else str(self.local_pred)
        }


class KernelShapExplainer:
    def __init__(self, path, random_state=42):
        self.random_state = check_random_state(random_state)
        self.base = KernelBase(random_state=self.random_state)
        self.path = path

    def explain_instance(self, label_to_explain=None, model_regressor=None, random_seed=42):
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        with open(self.path, 'r') as file:
            data = json.load(file)

        y = [score[label_to_explain] for score in data['scores']]
        y = np.array(y)
        
        distances = data['snrs']
        
        explanation = Explanation(np.array(data['snrs']), y)

        explanation.intercept, explanation.local_exp, explanation.score, explanation.local_pred = \
            self.base.explain_instance_with_data(
                np.array(data['snrs']), y, distances, model_regressor=model_regressor)


        return explanation
