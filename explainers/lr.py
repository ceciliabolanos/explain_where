import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.utils import check_random_state
from functools import partial
import json


class LRExplainer():

    def __init__(self, path, kernel_width=0.25):
        self.path = path
        self.kernel_width = kernel_width
        self.kernel_fn = partial(self.kernel, kernel_width=float(self.kernel_width))
    
    def kernel(d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    
    def get_feature_importances(self, 
                                label_to_explain,
                                weighting=None,
                                regularization=None):
        
        with open(self.path, 'r') as file:
            data = json.load(file)

        y = [score[label_to_explain] for score in data['scores']]
        y = np.array(y)
        Xs = np.array(data['snrs'])
        features = range(Xs.shape[1])

        if weighting:
            distances = data['neighborhood']
            min_non_zero_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else 1e-8
            distances = np.maximum(distances, min_non_zero_dist * 0.1)
            weights = self.kernel_fn(distances)
            weights = np.maximum(weights, 1e-8)
            weights = weights / np.sum(weights)
            weights = np.ones_like(weights)
        else:
            distances = data['neighborhood']
            weights = np.ones_like(distances)
    
        if regularization == 'lasso':
            model_regressor = Lasso(alpha=1e-7, fit_intercept=True, random_state=42)
        elif regularization == 'ridge':
            model_regressor = Ridge(alpha=1e-7, fit_intercept=True, random_state=42)
        elif regularization == 'noreg':
            model_regressor = LinearRegression(fit_intercept=True, random_state=42)
        
        model_regressor.fit(Xs[:, features], y, sample_weight=weights)
        local_pred = model_regressor.predict(Xs[0, features].reshape(1, -1))

        return model_regressor.coef_, local_pred