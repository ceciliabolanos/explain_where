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
        features = range(neighborhood_data.shape[1])

        X = neighborhood_data[1:, features]
        y = neighborhood_labels[1:]
        weights = np.array(weights)  # Ensure weights are in array format
        b_eq = [neighborhood_labels[0]]

        # Define objective function (Weighted Least Squares)
        def weighted_loss(coeffs):
            b0 = coeffs[-1]  # Last coefficient is b0
            phi_coeffs = coeffs[:-1]  # All other coefficients are phi values
            
            # Compute residuals including b0
            residuals = y - (np.dot(X, phi_coeffs) + b0)
            weighted_residuals = weights * residuals**2
            return np.sum(weighted_residuals)

        initial_coeffs = np.zeros(X.shape[1] + 1)  # Add one more dimension for b0
        constraint = {'type': 'eq', 'fun': lambda coeffs: np.sum(coeffs) - b_eq[0]}
        result = minimize(weighted_loss, initial_coeffs, constraints=constraint, method='SLSQP', options={'maxiter': 1000})
        
        if not result.success:
            raise RuntimeError(f"Optimization did not converge: {result.message}")
       
         
        b0 = result.x[-1]
        constrained_coeffs = result.x[:-1]
        local_pred = np.dot(neighborhood_data[0, features], constrained_coeffs) + b0
        
        return (b0,
                constrained_coeffs,
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


class KernelShapExplainer1constraint:
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
