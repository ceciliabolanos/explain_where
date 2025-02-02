
from .base_generator import BaseDataGenerator, MaskingConfig
import numpy as np
from typing import Optional, Callable
from scipy.spatial.distance import euclidean, cosine
from fastdtw import fastdtw 

class WindowMaskingDataGenerator(BaseDataGenerator):
    """Generator for audio data with masking windows"""
    def __init__(self,
                model_name,
                audio: np.ndarray,
                sample_rate: int,
                mask_config: MaskingConfig,
                predict_fn: Optional[Callable] = None):
        
        super().__init__(model_name=model_name,
                        config=mask_config,
                        input=audio,
                        predict_fn=predict_fn)
        self.sr = sample_rate

    
    def _generate_naive_masked(self):
        mask_samples = int(self.sr * (self.config.segment_length / 1000))
        windows = self.config.window_size
        overlap_samples = int(self.sr * (self.config.overlap / 1000))
        
        if mask_samples > len(self.input):
            raise ValueError("Mask duration is longer than audio length")
                
        results = []
        step_samples = mask_samples - overlap_samples
        
        current_pos = 0
        while current_pos < len(self.input):
            masked_audio = np.copy(self.input)
            
            end = min((current_pos + mask_samples) * windows, len(self.input))
            if self.config.mask_type == "zeros":
                masked_audio[current_pos:end] = 0

            elif self.config.mask_type == "noise":
                noise_std = np.random.uniform(0.01, 0.1)
                masked_audio[current_pos:end] = np.random.normal(np.mean(self.input), noise_std, end - current_pos)

            elif self.config.mask_type == "stat":
                fill_value = np.mean(self.input)
                masked_audio[current_pos:end] = fill_value 

            prediction = self.predict_fn([masked_audio])
            results.append(prediction[0]) #.cpu().detach().numpy().tolist()
    
            # Move to next position, starting the new mask before the current one ends
            current_pos += step_samples
                
        return results

    def segment_signal(self, S):
        """
        Split signal into overlapping segments
        
        Parameters:
        S: array-like, input signal
        L: int, segment length
        O: int, overlap size
        """
        W = [] 
        L = int((self.config.segment_length/1000) * self.sr)
        O = int((self.config.overlap/1000) * self.sr)
        for start in range(0, len(S) - L + 1, L - O):
            end = start + L
            if end <= len(S):
                W.append(S[start:end])

        last_start = len(W) * (L - O)
        if last_start < len(S):
            last_segment = S[last_start:]
            W.append(last_segment)
                
        return W

    def create_masked_input(self, row):
        W = self.segment_signal(self.input)
        L = int((self.config.segment_length / 1000) * self.sr)
        O = int((self.config.overlap / 1000) * self.sr)
        
        step = L - O
        output = np.copy(self.input)

        for i, (segment, use) in enumerate(zip(W, row)):
            start = i * step

            if not use:  # Apply masking
                end = start + step
                if i == len(W) - 1:  # Handle last segment to avoid index overflow
                    end = len(output)

                if self.config.mask_type == "zeros":
                    output[start:end] = 0

                elif self.config.mask_type == "noise":
                    noise_std = np.random.uniform(0.01, 0.1)
                    output[start:end] = np.random.normal(np.mean(self.input), noise_std, end - start)

                elif self.config.mask_type == "stat":
                    fill_value = np.mean(self.input)
                    output[start:end] = fill_value

                else:
                    raise ValueError("Invalid mask_type. Choose from 'zeros', 'noise', or 'stat'.")

        return output


    def _generate_all_masked(self):
        n_components = len(self.segment_signal(self.input))
        snrs = self.generate_specific_combinations(n_components=n_components, 
                                                   num_samples=self.config.num_samples, 
                                                   mask_percentage=self.config.mask_percentage,
                                                   window_size=self.config.window_size)
        
        scores, neighborhood = self.get_scores_neigh(batch_size=128, snrs=snrs)
        return scores, snrs, neighborhood
    
    def compute_similarity(self, temp):
        similarity_functions = {
            "euclidean": euclidean,
            "cosine": cosine,
            "dtw": lambda x, y: fastdtw(x, y)[0]  # DTW returns (distance, path), we take only distance
        }

        # Get the chosen function from config
        similarity_function = similarity_functions.get(self.config.function)

        return similarity_function(self.input, temp)
    
    def get_scores_neigh(self, batch_size=10, snrs=None):
        """Generates pertubed inputs from 1s and 0s and predictions in the neighborhood of this audio.
        Args:
            batch_size: classifier_fn will be called on batches of this size.
        """
        labels = []
        inputs_perturb = []
        neighborhood = []
        
        for row in snrs:
            temp = self.create_masked_input(row)
            inputs_perturb.append(temp)
            neighborhood.append(self.compute_similarity(temp))
            
            if len(inputs_perturb) == batch_size:
                preds = self.predict_fn(inputs_perturb)
                labels.extend(preds)
                inputs_perturb = []

        if len(inputs_perturb) > 0:
            preds = self.predict_fn(inputs_perturb)
            labels.extend(preds)
        
        return labels, neighborhood