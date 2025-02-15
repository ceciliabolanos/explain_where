import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from functools import partial
import json
from explainers.utils import compute_log_odds

class LimeBase:
    """Base class for learning locally linear sparse models from perturbed data."""
    
    def __init__(self, kernel_fn, verbose=False, absolute_feature_sort=False, random_state=None):
        """
        Initialize the LIME base explainer.
        
        Args:
            kernel_fn (callable): Function that transforms distances into proximity values.
            verbose (bool): If True, prints local prediction details.
            absolute_feature_sort (bool): Whether to sort features by absolute value.
            random_state (int or np.RandomState): Random number generator seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.absolute_feature_sort = absolute_feature_sort
        self.random_state = check_random_state(random_state)

    def explain_instance_with_data(self, neighborhood_data, neighborhood_labels, 
                                 distances, label, model_regressor=None, alpha=1):
        """
        Generate explanation for an instance using its neighborhood data.
        
        Args:
            neighborhood_data (np.ndarray): Perturbed data points, first row is the original instance.
            neighborhood_labels (np.ndarray): Labels for perturbed data points.
            distances (np.ndarray): Distances from original instance to perturbed instances.
            label (int): Label to explain.
            model_regressor (sklearn regressor, optional): Model for local approximation.
            alpha (float): Regularization strength for Ridge regression.
            
        Returns:
            tuple: (intercept, local_exp, prediction_score, local_pred)
                - intercept: Bias term of local model
                - local_exp: List of [coefficients, p-values]
                - prediction_score: R² score of local model
                - local_pred: Local model's prediction for original instance
        """
        # Calculate and normalize kernel weights
        weights = self.kernel_fn(distances)
        weights = np.maximum(weights, 1e-8)
        weights = weights / np.sum(weights)
        
        # Setup local model
        if model_regressor is None:
            model_regressor = Ridge(alpha=alpha, fit_intercept=True,
                                  random_state=self.random_state)
        
        # Fit local model
        features = range(neighborhood_data.shape[1])
        model_regressor.fit(neighborhood_data[:, features],
                          neighborhood_labels,
                          sample_weight=weights)
        
        # Evaluate local model
        prediction_score = model_regressor.score(
            neighborhood_data[:, features],
            neighborhood_labels,
            sample_weight=weights
        )
        local_pred = model_regressor.predict(neighborhood_data[0, features].reshape(1, -1))
        
        if self.verbose:
            print(f'Intercept: {model_regressor.intercept_}')
            print(f'Local prediction: {local_pred}')
            print(f'Actual: {neighborhood_labels[0]}')
            print(f'Score: {prediction_score}')
        
        return (model_regressor.intercept_,
                model_regressor.coef_,
                prediction_score,
                local_pred)


class Explanation:
    """Container for audio explanation results."""
    
    def __init__(self, neighborhood_data, neighborhood_labels):
        """
        Initialize audio explanation container.
        
        Args:
            neighborhood_data (np.ndarray): Perturbed instances data.
            neighborhood_labels (np.ndarray): Labels for perturbed instances.
        """
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


class LRExplainer:
    def __init__(self, path, kernel_width=0.25, kernel=None, verbose=False,
                 absolute_feature_sort=False, random_state=42):
        """
        Initialize audio explainer.
        
        Args:
            path (str): Path to data file.
            kernel_width (float): Width parameter for similarity kernel.
            kernel (callable, optional): Custom kernel function.
            verbose (bool): If True, prints additional information.
            absolute_feature_sort (bool): Whether to sort features by absolute value.
            random_state (int): Random seed for reproducibility.
        """
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
                 
        self.kernel_fn = partial(kernel, kernel_width=float(kernel_width))
        self.random_state = check_random_state(random_state)
        self.base = LimeBase(self.kernel_fn, verbose, absolute_feature_sort,
                           random_state=self.random_state)
        self.path = path

    def explain_instance(self, label_to_explain=None, model_regressor=None, random_seed=42, model='drums'):
        """
        Generate explanation for an audio instance.
        
        Args:
            label_to_explain (int, optional): Specific label to explain.
            model_regressor (sklearn regressor, optional): Local approximation model.
            alpha (float): Regularization strength.
            random_seed (int, optional): Random seed for reproducibility.
            
        Returns:
            AudioExplanation: Explanation object containing feature importances and predictions.
        """
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        # Load and prepare data
        with open(self.path, 'r') as file:
            data = json.load(file)

        
        if model == 'drums':
            y = compute_log_odds(data['scores'], label_to_explain)
            y = np.array(y)
        else:
            y = [score[label_to_explain] for score in data['scores']]
            y = np.array(y)
        distances = np.array(data['neighborhood'])
        
        # Handle potential zero distances
        min_non_zero_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else 1e-8
        distances = np.maximum(distances, min_non_zero_dist * 0.1)
        
        # Create and populate explanation object
        explanation = Explanation(np.array(data['snrs']), y)

        explanation.intercept, explanation.local_exp, explanation.score, explanation.local_pred = \
            self.base.explain_instance_with_data(
                np.array(data['snrs']), y, distances, label_to_explain,
                model_regressor=model_regressor, alpha=1e-7
            )


        return explanation
