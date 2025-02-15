import numpy as np
from sklearn.utils import check_random_state
import json
from math import comb
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from explainers.utils import compute_log_odds

def shap_kernel_weight(m, z):
    """
    Compute the raw Kernel SHAP weight for a coalition with z features present
    out of m total.
    
    For edge cases (z==0 or z==m) the true weight is infinite.
    """
    if z == 0 or z == m:
        return np.inf
    log_comb = gammaln(m + 1) - gammaln(z + 1) - gammaln(m - z + 1)
    log_weight = np.log(m - 1) - log_comb - np.log(z) - np.log(m - z)
    return np.exp(log_weight)


def pi_x_for_list(vectors):
    """
    Compute the Kernel SHAP weight for each vector in `vectors`.
    
    - If `normalize=True`, weights are divided by the weight for a coalition with z=1.
    - If `normalize_by_min=True`, weights are divided by the minimum nonzero weight.
    """
    weights = []
    for x in vectors:
        m = len(x)
        z = sum(x)
        weight = shap_kernel_weight(m, z)
        weights.append(weight)
    
    # if normalize_by_min:
    #     min_weight = min([w for w in weights if w > 0])  # Smallest nonzero weight
    #     weights = [w / min_weight for w in weights]
    # elif normalize:
    #     scaling = shap_kernel_weight(len(vectors[0]), 1)  # Normalize by z=1 weight
    #     weights = [w / scaling for w in weights]
    if len(set(weights)) == 1:
        weights = [1] * len(weights)
    mean = sum(weights)/len(weights)
    weights = [w / mean for w in weights]
    return weights


class KernelBase:
    def __init__(self, verbose=False, absolute_feature_sort=False, random_state=None):
        self.verbose = verbose
        self.absolute_feature_sort = absolute_feature_sort
        self.random_state = check_random_state(random_state)

    def explain_instance_with_data(self, neighborhood_data, neighborhood_labels, 
                                 distances, empty_score):
        
        weights = pi_x_for_list(distances[1:]) 

        b0 = empty_score
        features = range(neighborhood_data.shape[1])

        X = neighborhood_data[1:, features]
        y = neighborhood_labels[1:] - b0
        weights = np.array(weights)  
        b_eq = [neighborhood_labels[0]]

        # Define objective function (Weighted Least Squares)
        def weighted_loss(coeffs):
            residuals = y - np.dot(X, coeffs)  # Compute residuals
            weighted_residuals = weights * residuals**2  # Apply sample weights
            return np.sum(weighted_residuals)  # Minimize weighted sum of squared errors

        # Define constraint: sum(ϕ) + b0 = f_x
        constraint = {'type': 'eq', 'fun': lambda coeffs: np.sum(coeffs) + b0 - b_eq[0]}
        init_guess = np.zeros(X.shape[1])
        result = minimize(weighted_loss, init_guess, constraints=constraint, method='SLSQP', options={'maxiter': 3000})
        
        if not result.success:
            print(f"Optimization did not converge: {result.message, weights}")
            raise RuntimeError(f"Optimization did not converge: {result.message}")

        constrained_coeffs = result.x

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


class KernelShapExplainer:
    def __init__(self, path, random_state=42):
        self.random_state = check_random_state(random_state)
        self.base = KernelBase(random_state=self.random_state)
        self.path = path

    def explain_instance(self, label_to_explain=None, empty_score=None, random_seed=42, model='drums'):
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        with open(self.path, 'r') as file:
            data = json.load(file)

        if model == 'drums':
            y = compute_log_odds(data['scores'], label_to_explain)
            y = np.array(y)
        else:
            y = [score[label_to_explain] for score in data['scores']]
            y = np.array(y)
        distances = data['snrs']
        
        explanation = Explanation(np.array(data['snrs']), y)

        explanation.intercept, explanation.local_exp, explanation.local_pred = \
            self.base.explain_instance_with_data(
                np.array(data['snrs']), y, distances, empty_score)


        return explanation
