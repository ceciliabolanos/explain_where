import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
import json
from math import comb
import numpy as np
from scipy.optimize import minimize

def pi_x_for_list(vectors):
    results = []
    for x in vectors:
        m = len(x)
        z = sum(x) 
        if z == 0 or z == m:
            results.append(10000) 
            continue

        binom_mz = comb(m, z)
        result = (m - 1) / (binom_mz * z * (m - z))
        results.append(result)

    mean_result = sum(results) / len(results) if results else 1  # Avoid division by zero
    normalized_results = [r / mean_result for r in results]
    return normalized_results

class KernelBase:
    def __init__(self, verbose=False, absolute_feature_sort=False, random_state=None):
        self.verbose = verbose
        self.absolute_feature_sort = absolute_feature_sort
        self.random_state = check_random_state(random_state)

    def explain_instance_with_data(self, neighborhood_data, neighborhood_labels, 
                                 distances, empty_score):
        
        weights = pi_x_for_list(distances[1:]) 
        # model_regressor = LinearRegression(fit_intercept=True)

        # # Fit local model
        # features = range(neighborhood_data.shape[1])
        # model_regressor.fit(neighborhood_data[1:, features], 
        #                     neighborhood_labels[1:], 
        #                     sample_weight=weights)
        
        # local_pred = model_regressor.predict(neighborhood_data[0, features].reshape(1, -1))
        
        b0 = empty_score
        features = range(neighborhood_data.shape[1])

        X = neighborhood_data[1:, features]
        y = neighborhood_labels[1:] 
        weights = np.array(weights)  # Ensure weights are in array format
        b_eq = [neighborhood_labels[0]]
        # Define objective function (Weighted Least Squares)
        def weighted_loss(coeffs):
            residuals = y - np.dot(X, coeffs)  # Compute residuals
            weighted_residuals = weights * residuals**2  # Apply sample weights
            return np.sum(weighted_residuals)  # Minimize weighted sum of squared errors

        # Define constraint: sum(ϕ) + b0 = f_x
        constraint = {'type': 'eq', 'fun': lambda coeffs: np.sum(coeffs) + b0 - b_eq[0]}
        init_guess = np.zeros(X.shape[1])
        result = minimize(weighted_loss, init_guess, constraints=constraint, method='SLSQP')
        constrained_coeffs = result.x

        local_pred = np.dot(neighborhood_data[0, features], constrained_coeffs) + b0

        
        return (b0,
                constrained_coeffs,
                local_pred)

        # return (model_regressor.intercept_,
        #         model_regressor.coef_,
        #         local_pred)
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

    def explain_instance(self, label_to_explain=None, empty_score=None, random_seed=42):
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        with open(self.path, 'r') as file:
            data = json.load(file)

        y = [score[label_to_explain] for score in data['scores']]
        y = np.array(y)
        
        distances = data['snrs']
        
        explanation = Explanation(np.array(data['snrs']), y)

        explanation.intercept, explanation.local_exp, explanation.local_pred = \
            self.base.explain_instance_with_data(
                np.array(data['snrs']), y, distances, empty_score)


        return explanation
