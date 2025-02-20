import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
import json
from math import comb
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

class SHAPExplainer():

    def __init__(self, path):
        self.path = path

    def shap_kernel_weight(self, m, z):
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
   
    def pi_x_for_list(self, vectors):
        """
        Compute the Kernel SHAP weight for each vector in `vectors`.
        
        - If `normalize=True`, weights are divided by the weight for a coalition with z=1.
        - If `normalize_by_min=True`, weights are divided by the minimum nonzero weight.
        """
        weights = []
        for x in vectors:
            m = len(x)
            z = sum(x)
            weight = self.shap_kernel_weight(m, z)
            weights.append(weight)
        
        if len(set(weights)) == 1:
            weights = [1] * len(weights)
        mean = sum(weights)/len(weights)
        weights = [w / mean for w in weights]
        return weights


    def get_feature_importances(self, 
                                label_to_explain,
                                weighting='shap_values',
                                empty_constraint=None):
        
        with open(self.path, 'r') as file:
            data = json.load(file)

        y = [score[label_to_explain] for score in data['scores']]
        y = y[1:]
        y = np.array(y)

        Xs = np.array(data['snrs'])
        X = Xs[1:, features]
        features = range(Xs.shape[1])

        if weighting:
            distances = data['snrs']
            weights = self.pi_x_for_list(distances[1:]) # dont use the first instance because its the original one    
            weights = np.array(weights)  # Ensure weights are in array format
        
        b_eq = y[0]

        # Define objective function (Weighted Least Squares)
        def weighted_loss_1cons(coeffs):
            b0 = coeffs[-1]  # Last coefficient is b0
            phi_coeffs = coeffs[:-1]  # All other coefficients are phi values
            
            # Compute residuals including b0
            residuals = y - (np.dot(X, phi_coeffs) + b0)
            weighted_residuals = weights * residuals**2
            return np.sum(weighted_residuals)
        
        def weighted_loss(coeffs):
            residuals = y - np.dot(X, coeffs)  # Compute residuals
            weighted_residuals = weights * residuals**2  # Apply sample weights
            return np.sum(weighted_residuals)  # Minimize weighted sum of squared errors
        
        if empty_constraint:
            # Define constraint: sum(ϕ) + empty_constraint = f_x
            y = y - empty_constraint
            init_guess = np.zeros(X.shape[1])
            constraint = {'type': 'eq', 'fun': lambda coeffs: np.sum(coeffs) + empty_constraint - b_eq}
            result = minimize(weighted_loss, init_guess, constraints=constraint, method='SLSQP', options={'maxiter': 3000})
            constrained_coeffs = result.x
            local_pred = np.dot(Xs[0, features], constrained_coeffs) + empty_constraint
       
        else:
            initial_coeffs = np.zeros(X.shape[1] + 1)  # Add one more dimension for b0
            constraint = {'type': 'eq', 'fun': lambda coeffs: np.sum(coeffs) - b_eq}
            result = minimize(weighted_loss_1cons, initial_coeffs, constraints=constraint, method='SLSQP', options={'maxiter': 2000})
            b0 = result.x[-1]
            constrained_coeffs = result.x[:-1]
            local_pred = np.dot(Xs[0, features], constrained_coeffs) + b0
        
        if not result.success:
            print(f"Optimization did not converge:  {result.message, weights}")
            raise RuntimeError(f"Optimization did not converge: {result.message}")
       
        return constrained_coeffs, local_pred