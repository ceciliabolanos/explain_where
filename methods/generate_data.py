import numpy as np
import os 
import json 
from scipy.spatial.distance import euclidean
import random
import math
from itertools import combinations

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class DataGenerator:
    def __init__(self, wav, 
                 mode='naive_masked_zeros', 
                 segment_length=100, 
                 mask_percentage=0.3, 
                 window_size=3, 
                 num_samples=10, 
                 predict_fn=None, 
                 sr=16000):
        
        self.mode = mode
        self.segment_length = segment_length
        self.mask_percentage = mask_percentage
        self.window_size = window_size
        self.overlap = 0
        self.num_samples = num_samples
        self.predict_fn = predict_fn
        self.sr = sr
        self.wav = wav

    def generate(self, filename):
        # Elegimos una ventana y le ponemos valor cero 
        if self.mode == 'naive_masked_zeros':
            data_to_save = {
            "scores": self._generate_naive_masked(),
            "neighborhood": None,
            "score_real": self.predict_fn(self.wav),
            "snrs" : None
        }
        
        # Elegimos al azar una cantidad de ventanas que sumamos ruido
        if self.mode == 'all_noise':
            scores, snrs, neighborhood = self._generate_all_noise()
        
            data_to_save = {
                "scores": scores,
                "neighborhood": neighborhood.tolist(),
                "score_real": self.predict_fn(self.wav),
                "snrs" : snrs
            }
            

        # Elegimos al azar una cantidad de ventanas que enmascaramos
        if self.mode == 'all_masked':
            scores, snrs, neighborhood = self._generate_all_masked()
        
            data_to_save = {
                "scores": scores,
                "neighborhood": neighborhood,
                "score_real": self.predict_fn(self.wav),
                "snrs" : snrs
            }
        if self.mode == 'all_masked':
            output_file = f"/home/cbolanos/experiments/audioset_audios_eval/{filename}/scores_data_{self.mode}_p{self.mask_percentage}_m{self.window_size}.json"
        else:
            output_file = f"/home/cbolanos/experiments/audioset_audios_eval/{filename}/scores_data_{self.mode}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as json_file:
            json.dump(data_to_save, json_file)    

    def _generate_naive_masked(self):
        # Calculate sizes in samples
        mask_samples = int(self.sr * (self.segment_length / 1000))
        overlap_samples = int(self.sr * (self.overlap / 1000))
        
        if mask_samples > len(self.wav):
            raise ValueError("Mask duration is longer than audio length")
                
        results = []

        # Calculate step size (distance between start of consecutive masks)
        step_samples = mask_samples - overlap_samples
        
        current_pos = 0
        while current_pos < len(self.wav):
            masked_audio = np.copy(self.wav)
            
            # Calculate end point for current mask
            end = min(current_pos + mask_samples, len(self.wav))
            
            # Apply mask
            masked_audio[current_pos:end] = 0
            
            prediction = self.predict_fn(masked_audio)
            results.append(prediction)
            # Move to next position, starting the new mask before the current one ends
            current_pos += step_samples
            
        if not results:
            raise RuntimeError("No valid predictions were generated")
                
        return results
    
    def _generate_all_noise(self):
        NewData, snrs = self.data_generator(self.num_samples, self.wav, self.segment_length, self.overlap, SNR)
        all_scores = []
        neighborhood = np.zeros(self.num_samples + 1)

        all_scores.extend(self.predict_fn(self.wav))
        neighborhood[0] = euclidean(self.wav, self.wav)

        for i in range(len(NewData)):
            scores = self.predict_fn(NewData[i,:])
            neighborhood[i+1] = euclidean(self.wav, NewData[i, :])
            all_scores.extend(scores) 

        tamaño = len(snrs[0])  # Tamaño del elemento en la lista
        # Crea el nuevo elemento con ceros y agrégalo al principio
        nuevo_elemento = [0] * tamaño
        snrs.insert(0, nuevo_elemento)

        return all_scores, snrs, neighborhood


    def segment_signal(self, S):
        """
        Split signal into overlapping segments
        
        Parameters:
        S: array-like, input signal
        L: int, segment length
        O: int, overlap size
        """
        W = [] 
        L = int((self.segment_length/1000) * self.sr)
        O = int(( self.overlap/1000) * self.sr)
        for start in range(0, len(S) - L + 1, L - O):
            end = start + L
            if end <= len(S):
                W.append(S[start:end])

        last_start = len(W) * (L - O)
        if last_start < len(S):
            last_segment = S[last_start:]
            W.append(last_segment)
                
        return W


    def generate_noise_for_segment(self, segment, snr):
        """
        Generate Gaussian noise for a signal segment based on SNR.
        
        Parameters:
        segment: array-like, the signal segment
        snr: float, Signal-to-Noise Ratio in dB
        
        Returns:
        array-like: Generated noise matching segment length
        """
        # Calculate signal power
        signal_power = np.sum(np.square(segment)) / len(segment)
        
        # Calculate noise power based on the formula: Pn = signal_power / (10^(SNR/10))
        noise_power = signal_power / (10 ** (snr / 10))
        
        # Generate noise matching segment length exactly
        noise = np.random.normal(0, np.sqrt(noise_power), size=len(segment))
        
        return noise


    def data_generator(self, Z, S, L, O, SNR, noise_prob=0.5):
        """
        Generate perturbed versions of the signal with probabilistic noise addition
        
        Parameters:
        Z: int, Number of samples to generate
        S: array-like, Original signal
        L: int, Segment length
        O: int, Overlap
        SNR: list, Signal-to-noise ratios to use
        noise_prob: float, Probability of adding noise to each segment (between 0 and 1)
        """
        # Input validation
        if not 0 <= noise_prob <= 1:
            raise ValueError("noise_prob must be between 0 and 1")
        
        # Get segments
        W = self.segment_signal(S)
        print(len(W))
        NF = len(S)
        NewData = np.zeros((Z, NF))
        snrs = []
        for z in range(Z):
            # Start with clean signal
            perturbed_signal = np.copy(S)
            sample_snrs = []
            current_pos = 0
            for segment in W:
                # Decide whether to add noise to this segment
                if np.random.random() < noise_prob:
                    # Select random SNR
                    snr = np.random.choice(SNR)
                    
                    # Generate noise matching current segment length
                    noise = self.generate_noise_for_segment(segment, snr)
                    
                    # Calculate end position
                    end_pos = min(current_pos + len(segment), len(S))
                    segment_length = end_pos - current_pos
                    
                    # Add noise to the corresponding position
                    perturbed_signal[current_pos:end_pos] += noise[:segment_length]
                else: 
                    snr = 0
                sample_snrs.append(int(snr))
                # Update position
                current_pos += int((self.segment_length/1000) * self.sr) - int(( self.overlap/1000) * self.sr)
                if current_pos >= len(S):
                    break
                    
            NewData[z] = perturbed_signal
            snrs.append(sample_snrs)
        
        return NewData, snrs        

    def _generate_all_masked(self):
        n_components = len(self.segment_signal(self.wav))
        snrs = self.generate_specific_combinations(n_components=n_components, 
                                                   num_samples=self.num_samples, 
                                                   mask_percentage=self.mask_percentage,
                                                   window_size=self.window_size)
        
        scores, neighborhood = self.get_scores_neigh(batch_size=256, snrs=snrs)
        return scores, snrs, neighborhood
    
    def create_masked_wav(self, row):
        W = self.segment_signal(self.wav)
        L = int((self.segment_length/1000) * self.sr)
        O = int((self.overlap/1000) * self.sr)

        step = L - O
        output = np.zeros(len(self.wav))

        for i, (segment, use) in enumerate(zip(W, row)):
            if use:
                start = i * step
                if i == len(W) - 1:
                    # Get the remaining length
                    remaining_length = len(output) - start
                    # Copy only up to the remaining length
                    output[start:start+remaining_length] = segment[:remaining_length]
                else:
                    # Normal case - copy the whole segment
                    output[start:start+step] = segment[:step]
        return output

    def get_scores_neigh(self, batch_size=10, snrs=None):
        """Generates audio and predictions in the neighborhood of this audio.
        Args:
            batch_size: classifier_fn will be called on batches of this size.
        """
        labels = []
        audios = []
        neighborhood = []
        j=0
        for row in snrs:
            temp = self.create_masked_wav(row)
            audios.append(temp)
            neighborhood.append(euclidean(self.wav, temp))
            if len(audios) == batch_size:
                preds = self.predict_fn(np.array(audios))
                labels.extend(preds)
                audios = []
        if len(audios) > 0:
            preds = self.predict_fn(np.array(audios))
            labels.extend(preds)
        return  labels, neighborhood

    # def generate_specific_combinations(self, n_components):
    #     all_combinations = [np.ones(n_components, dtype=int).tolist()]  # Start with all-ones array
    #     max_zeros = math.ceil(n_components * 0.30)  # 30% redondeado hacia arriba
        
    #     for num_zeros in range(1, max_zeros + 1):
    #         zero_positions = list(combinations(range(n_components), num_zeros))
            
    #         for positions in zero_positions:
    #             if len(all_combinations) >= self.num_samples:  
    #                 return all_combinations
                
    #             arr = np.ones(n_components, dtype=int)
    #             arr[list(positions)] = 0
    #             all_combinations.append(arr.tolist())
        
    #     return all_combinations

    def generate_masked_combinations(self, n_components, mask_percentage=0.3, window_size=3):
        """
        Generate masked combinations using window-based approach.
        
        Args:
            n_components (int): Total number of components
            mask_percentage (float): Percentage of audio to mask (0-100)
            window_size (int): Size of masking window
        
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
            # Select a random starting position
            start_pos = random.choice(possible_positions)
            possible_positions.remove(start_pos)
            
            # Check how many unmasked positions we would actually mask
            effective_mask = 0
            for i in range(start_pos, min(start_pos + window_size, n_components)):
                if result[i] == 1:
                    effective_mask += 1
            
            # If adding this window would exceed target, try to find a better position
            if total_masked + effective_mask > target_masked:
                continue
                
            # Apply the mask
            result[start_pos:start_pos + window_size] = 0
            total_masked += effective_mask
            selected_positions.append(start_pos)
        
        # If we haven't reached target_masked, try to add individual positions
        if total_masked < target_masked:
            remaining = target_masked - total_masked
            for i in range(n_components):
                if remaining <= 0:
                    break
                if result[i] == 1:
                    result[i] = 0
                    remaining -= 1
                    
        return result.tolist()

    def generate_specific_combinations(self, n_components, num_samples, mask_percentage=0.3, window_size=3):
        """
        Generate multiple masked combinations.
        
        Args:
            n_components (int): Total number of components
            num_samples (int): Number of combinations to generate
            mask_percentage (float): Percentage of audio to mask (0-100)
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