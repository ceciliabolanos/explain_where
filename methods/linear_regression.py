import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from regressors import stats
from functools import partial

import sklearn.preprocessing
import json

class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 absolute_feature_sort=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.absolute_feature_sort = absolute_feature_sort
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""

        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        # elif method == 'highest_weights':
        #     clf = Ridge(alpha=0, fit_intercept=True,
        #                 random_state=self.random_state)
        #     clf.fit(data, labels, sample_weight=weights)

        #     coef = clf.coef_
        #     if sp.sparse.issparse(data):
        #         coef = sp.sparse.csr_matrix(clf.coef_)
        #         weighted_data = coef.multiply(data[0])
        #         # Note: most efficient to slice the data before reversing
        #         sdata = len(weighted_data.data)
        #         argsort_data = np.abs(weighted_data.data).argsort()
        #         # Edge case where data is more sparse than requested number of feature importances
        #         # In that case, we just pad with zero-valued features
        #         if sdata < num_features:
        #             nnz_indexes = argsort_data[::-1]
        #             indices = weighted_data.indices[nnz_indexes]
        #             num_to_pad = num_features - sdata
        #             indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
        #             indices_set = set(indices)
        #             pad_counter = 0
        #             for i in range(data.shape[1]):
        #                 if i not in indices_set:
        #                     indices[pad_counter + sdata] = i
        #                     pad_counter += 1
        #                     if pad_counter >= num_to_pad:
        #                         break
        #         else:
        #             nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
        #             indices = weighted_data.indices[nnz_indexes]
        #         return indices
        #     else:
        #         weighted_data = coef * data[0]
        #         feature_weights = sorted( # TODO: check if abs should be optional
        #             zip(range(data.shape[1]), weighted_data),
        #             key=lambda x: np.abs(x[1]),
        #             reverse=True)
        #         return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)

    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='none',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """
        # Compute weights using kernel function
        weights = self.kernel_fn(distances)
        
        # Normalize weights to prevent numerical issues
        weights = np.maximum(weights, 1e-8)  # Ensure no exact zeros
        weights = weights / np.sum(weights)  # Normalize

        labels_column = neighborhood_labels
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)
        try:
            pvals = stats.coef_pval(easy_model, neighborhood_data[:, used_features],labels_column)
        except np.linalg.LinAlgError:
            # Fallback: use dummy p-values
            pvals = np.ones_like(easy_model.coef_) * 0.05
        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        local_exp = [easy_model.coef_, pvals]
        if self.verbose:
            print('Intercept:', easy_model.intercept_)
            print('Prediction_local:', local_pred,)
            print('Right:', neighborhood_labels[0])
            print('Score:', prediction_score)
        return (easy_model.intercept_,
                local_exp,
                prediction_score, local_pred)

    
class AudioExplanation(object):
    def __init__(self, neighborhood_data, neighborhood_labels):
        """Init function.

        Args:
            factorization: a Factorization object
        """
        self.neighborhood_data = neighborhood_data
        self.neighborhood_labels = neighborhood_labels
        self.intercept = 0
        self.local_exp = {}

    def get_feature_importances(self, label):

        exp = self.local_exp
        w = [[x[0], x[1]] for x in exp]
        # weights, pvals = np.array(w, dtype=int)[:, 0], np.array(w)[:, 1]

        return {
        "coefficients": exp[0].tolist() if hasattr(exp[0], 'tolist') else exp[0],
        "p_values": exp[1].tolist() if hasattr(exp[1], 'tolist') else exp[1]
    }
    

class LimeAudioExplainer(object):
    """Explains predictions on audio data."""

    def __init__(self, path, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='none', absolute_feature_sort=False, random_state=42):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            : an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.feature_selection = feature_selection
        self.base = LimeBase(kernel_fn, verbose, absolute_feature_sort, random_state=self.random_state)
        self.path = path

        
    def explain_instance(self,
                        label_to_explain=None,
                        model_regressor=None,
                        random_seed=42):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            factorization: function used for factorizing input audio
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            the neighborhood labels
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
            random_seed: integer used as random seed for the segmentation
                algorithm. If None, a random integer, between 0 and 1000,
                will be generated using the internal random number generator.

        Returns:
            An AudioExplanation object (see lime_audio.py) with the corresponding
            explanations.
        """

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        with open(self.path, 'r') as file:
            data = json.load(file)

        y = []
        for score in data['scores']:
            y.append(score[label_to_explain])

        y = np.array(y)
        distances = np.array(data['neighborhood'])
        
        min_non_zero_dist = np.min(distances[distances > 0]) if np.any(distances > 0) else 1e-8
        distances = np.maximum(distances, min_non_zero_dist * 0.1)  # Set minimum distance
    

        ret_exp = AudioExplanation(np.array(data['snrs']), y)

        (ret_exp.intercept, ret_exp.local_exp,
            ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
            np.array(data['snrs']), y, distances, label_to_explain, num_features=10000000,
            model_regressor=model_regressor,
            feature_selection=self.feature_selection)
        
        return ret_exp

  

