import numpy as np
import os 
import json 
import random
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class MaskingConfig:
    """Configuration for masking parameters."""

    def __init__(
        self,
        mask_percentage: float = 0.4,
        window_size: int = 3,
        num_samples: int = 3000,
        overlap: int = 0,
        segment_length: int = 100,
        function: str = 'euclidean',
        mask_type: str = 'noise',
        std_noise: float = None
    ):
        self.mask_percentage = mask_percentage
        self.window_size = window_size
        self.num_samples = num_samples
        self.overlap = overlap
        self.segment_length = segment_length
        self.function = function
        self.mask_type = mask_type
        self.std_noise = std_noise

class BaseDataGenerator(ABC):
    def __init__(self, 
                 model_name,
                 mode: str = 'all_masked',
                 config: MaskingConfig = None,
                 input: Any = None, 
                 predict_fn: Any = None,
                 filename: str = None,
                 id_to_explain: int = None,
                 path: str = None):
        """
        Base class for data perturbation generators.
        
        Args:
            model: Model that we want to explain
            mode: Masking strategy to use
            mask_config: Configuration for the masking algorithm
            predict_fn: Function to make predictions on perturbed input
        """
        self.model_name = model_name
        self.mode = mode
        self.config = config
        self.predict_fn = predict_fn
        self.input = input
        self.filename = filename
        self.id_to_explain = id_to_explain
        self.path = path

    @abstractmethod
    def _generate_naive_masked(self) -> Any:
        """
        Generates a naive mask by masking one element at a time based on the selected measure 
        (e.g., token, word, or a specific window).
        """
        pass

    @abstractmethod
    def _generate_all_masked(self) -> Any:
        """A naive masking approach masked one at the time the measure selected (e.g token, word, specific window)"""
        pass

    @abstractmethod
    def create_masked_input(self) -> Any:
        """
        Given a list of 1s and 0s we generate the input masked 
        (it can be the waveform masked or the inputs_id masked).
        """
        pass

    def generate(self, filename):
        if self.mode == 'naive_masked':
            data_to_save = {
            "scores": self._generate_naive_masked(),
            "neighborhood": None,
            "score_real": self.predict_fn([self.input])[0],
            "snrs" : None
        }

        # Elegimos al azar una cantidad de ventanas que enmascaramos
        if self.mode == 'all_masked':
            scores, snrs, neighborhood = self._generate_all_masked(self.filename)
        
            data_to_save = {
                "scores": scores,
                "neighborhood": neighborhood,
                "score_real": self.predict_fn([self.input])[0],
                "snrs" : snrs
            }

        if self.mode == 'all_masked':
            if self.config.mask_type == 'zeros':
                output_file = Path(self.path) / filename / self.model_name / f"scores_p{self.config.mask_percentage}_w{self.config.window_size}_f{self.config.function}_m{self.config.mask_type}.json"
            elif self.config.mask_type == 'noise':
                output_file = Path(self.path) / filename / self.model_name / f"scores_p{self.config.mask_percentage}_w{self.config.window_size}_f{self.config.function}_m{self.config.mask_type}_a{self.config.std_noise}.json"
        else:
            output_file = Path(self.path) / filename / self.model_name / f"scores_w{self.config.window_size}_m{self.config.mask_type}.json"
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as json_file:
            json.dump(data_to_save, json_file)    
    

    def generate_masked_combinations(self, n_components, mask_percentage=0.3, window_size=3):
        """
        Generate masked combinations, the only thing important in here is to defined how
        many segments are we going to masked.
        
        Args:
            n_components (int): Total number of components
            mask_percentage (float): Percentage to mask (0-1)
            window_size (int): Size of masking window together
        
        Returns:
            list: Array of 1s and 0s representing the masking pattern
        """
        result = np.ones(n_components, dtype=int)
        
        # Calculate target number of components to mask
        target_masked = int(np.ceil(n_components * mask_percentage))
        total_masked = 0
        
        # Generate initial set of random positions
        possible_positions = list(range(n_components + 1))
        selected_positions = []
        
        while total_masked < target_masked and possible_positions:
            start_pos = random.choice(possible_positions)
            possible_positions.remove(start_pos)
            
            effective_mask = 0
            for i in range(start_pos, min(start_pos + window_size, n_components)):
                if result[i] == 1:
                    effective_mask += 1
            
            if total_masked + effective_mask > target_masked:
                continue
                
            result[start_pos:start_pos + window_size] = 0
            total_masked += effective_mask
            selected_positions.append(start_pos)
        
        if total_masked < target_masked:
            remaining = target_masked - total_masked
            while remaining > 0:
                for i in range(n_components):
                    if remaining <= 0:
                        break
                    if result[i] == 1:  # Expand around existing masked regions
                        if i > 0 and result[i - 1] == 0:
                            result[i - 1] = 1
                            remaining -= 1
                        if i < n_components - 1 and result[i + 1] == 0 and remaining > 0:
                            result[i + 1] = 1
                            remaining -= 1

        return result.tolist()

    def generate_specific_combinations(self, n_components, num_samples, mask_percentage=0.3, window_size=3):
        """
        Generate multiple masked combinations.
        
        Args:
            n_components (int): Total number of components
            num_samples (int): Number of combinations to generate
            mask_percentage (float): Percentage of input to mask (0-1)
            window_size (int): Size of masking window
        
        Returns:
            list: List of masked combinations
        """
        combinations = [np.ones(n_components, dtype=int).tolist()]
        for _ in range(num_samples):
            combination = self.generate_masked_combinations(
                n_components,
                mask_percentage,
                window_size
            )
            combinations.append(combination)
        return combinations