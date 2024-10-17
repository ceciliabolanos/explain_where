"""
Functions for explaining classifiers that use Image data.
"""
from functools import partial
from itertools import combinations
import math

import os
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state

from audioLIME import lime_base

class AudioExplanation(object):
    def __init__(self, factorization, factors, neighborhood_data, neighborhood_labels):
        """Init function.

        Args:
            factorization: a Factorization object
        """
        self.factorization = factorization
        self.components = factors
        self.neighborhood_data = neighborhood_data
        self.neighborhood_labels = neighborhood_labels
        self.intercept = {}
        self.local_exp = {}
        self.local_pred = None
        self.distance = {}

    def get_sorted_components(self, label, positive_components=True, negative_components=True, num_components='auto',
                              thresh=1e-8, min_abs_weight=0.0, return_indeces=False):
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_components is False and negative_components is False:
            raise ValueError('positive_components, negative_components or both must be True')

        exp = self.local_exp[label]
        print(f'this is the label {label}')
        w = [[x[0], x[1], x[2]] for x in exp]
        used_features, weights, pvals = np.array(w, dtype=int)[:, 0], np.array(w)[:, 1], np.array(w)[:, 2]

        if not negative_components:
            pos_weights = np.argwhere(weights > 0)[:, 0]
            used_features = used_features[pos_weights]
            weights, pvals = weights[pos_weights], pvals[pos_weights]
        elif not positive_components:
            neg_weights = np.argwhere(weights < 0)[:, 0]
            used_features = used_features[neg_weights]
            weights, pvals = weights[neg_weights], pvals[neg_weights]
        if min_abs_weight != 0.0:
            abs_weights = np.argwhere(abs(weights) >= min_abs_weight)[:, 0]
            used_features = used_features[abs_weights]
            weights, pvals = weights[abs_weights], pvals[abs_weights]

        if num_components == 'auto':
            # Auto thresholding as introduced in [Haunschmid, Chowdhury and Widmer 2019]
            thresh = np.abs(thresh)  # just to make sure
            auto_weights = np.where(((weights < 0) & (pvals / weights > -thresh)) |
                                    ((weights > 0) & (pvals / weights < thresh)))[0]
            used_features = used_features[auto_weights]
            num_components = len(used_features)
        elif num_components == 'all':
            num_components = len(used_features)
        else:
            assert(isinstance(num_components, int))
            # max_components = used_features[:num_components]

        used_features = used_features[:num_components]
        components = self.factorization.retrieve_components(used_features)
        if return_indeces:
            return components, used_features
        return components
    

    def get_components_with_stats(self, label, positive_components=True, negative_components=True, num_components='auto',
                              thresh=1e-8, min_abs_weight=0.0):
        # First, get the components and indices using the original function
        if label not in self.local_exp:
            raise KeyError('Label not in explanation')
        if positive_components is False and negative_components is False:
            raise ValueError('positive_components, negative_components or both must be True')

        exp = self.local_exp[label]
        print(f'this is the label {label}')
        w = [[x[0], x[1], x[2]] for x in exp]
        used_features, weights, pvals = np.array(w, dtype=int)[:, 0], np.array(w)[:, 1], np.array(w)[:, 2]

        return used_features, weights, pvals
    
class LimeAudioExplainer(object):
    """Explains predictions on audio data."""

    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', absolute_feature_sort=False, random_state=None):
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
        self.base = lime_base.LimeBase(kernel_fn, verbose, absolute_feature_sort, random_state=self.random_state)

    def explain_instance(self, factorization, predict_fn,
                         labels=None,
                         top_labels=None,
                         num_reg_targets=None,
                         num_features=100000,
                         num_samples=1000,
                         batch_size=10,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            factorization: function used for factorizing input audio
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities.  For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            batch_size: nr. of samples passed to the global model per batch when computing
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

        # check whether regression or classification task
        is_classification = False
        if labels or top_labels:
            is_classification = True
        if is_classification and num_reg_targets:
            raise ValueError('Set labels or top_labels for classification. '
                             'Set num_reg_targets for regression.')

        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        self.factorization = factorization
        try:
            factors = factorization.retrieve_components()
        except ValueError as e:
            raise e

        top = labels

        data, labels = self.data_labels(predict_fn, num_samples,
                                        batch_size=batch_size)
        output_dir = f"/home/cbolanos/experiments/emotion/Ses02F_script01_1_M013.wav"
        os.makedirs(output_dir, exist_ok=True)

        # Assuming 'data' is the matrix to save
        csv_path = os.path.join(output_dir, "data.csv")
        np.savetxt(csv_path, data, delimiter=",", fmt="%s")

        # If you also want to save labels:
        labels_path = os.path.join(output_dir, "labels.csv")
        np.savetxt(labels_path, labels, delimiter=",", fmt="%s")
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        ret_exp = AudioExplanation(self.factorization, factors, data, labels)

        if is_classification:
            if top_labels:
                top = np.argsort(labels[0])[-top_labels:]
                ret_exp.top_labels = list(top)
                ret_exp.top_labels.reverse()
            for label in top:
                (ret_exp.intercept[label],
                 ret_exp.local_exp[label],
                 ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                    data, labels, distances, label, num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    predict_fn,
                    num_samples,
                    batch_size=10):
        """Generates audio and predictions in the neighborhood of this audio.

        Args:
            predict_fn: function that takes a list of audio inputs and returns a
                matrix of predictions
            num_samples: size of the neighborhood to learn the linear model
            batch_size: classifier_fn will be called on batches of this size.

        Returns:
            A tuple (data, labels), where:
                data: dense num_samples * num_factors
                labels: prediction probabilities matrix
        """
        n_features = self.factorization.get_number_components()
        if num_samples == 'exhaustive':
            data = self.generate_specific_zero_combinations(n_features)
            num_samples = len(data)
        else:
            data = self.generate_random_masked_data(num_samples, n_features)
        print(f'number of components {n_features} generated tantos instancias {len(data)}')
        labels = []
        audios = []
        for row in data:
            non_zeros = np.where(row != 0)[0]
            if len(non_zeros) == 0:
                like = self.factorization.compose_model_input()
                temp = np.zeros_like(like) + like.min()
            else:
                temp = self.factorization.compose_model_input(non_zeros)
            audios.append(temp)
            if len(audios) == batch_size:
                preds = predict_fn(np.array(audios))
                labels.extend(preds)
                audios = []
        if len(audios) > 0:
            preds = predict_fn(np.array(audios))
            labels.extend(preds)
        return data, np.array(labels)

    def generate_specific_zero_combinations(self, n_components):
        all_combinations = [np.ones(n_components, dtype=int)]  # Start with all-ones array
        max_zeros = math.ceil(n_components * 0.30)  # 30% redondeado hacia arriba
        
        for num_zeros in range(1, max_zeros + 1):
            zero_positions = list(combinations(range(n_components), num_zeros))
            
            for positions in zero_positions:
                if len(all_combinations) >= 10000:  # Check if we've reached 10,000 samples
                    return  np.array(all_combinations)
                
                arr = np.ones(n_components, dtype=int)
                arr[list(positions)] = 0
                all_combinations.append(arr)
        
        return np.array(all_combinations)

    def generate_random_masked_data(self, num_samples, n_features):
        data = np.ones((num_samples, n_features), dtype=int)
        
        for i in range(1, num_samples):  # Start from second row
            num_zeros = self.random_state.choice([1, 2, 3])
            zero_indices = self.random_state.choice(n_features, num_zeros, replace=False)
            data[i, zero_indices] = 0