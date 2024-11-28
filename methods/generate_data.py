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
    def __init__(self, wav, mode='naive_masked_zeros', segment_length=500, overlap=100, num_samples=10, predict_fn=None, sr=16000):
        self.mode = mode
        self.segment_length = segment_length
        self.overlap = overlap
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
        snrs = self.generate_specific_combinations(n_components=n_components)
        scores, neighborhood = self.get_scores_neigh(batch_size=400, snrs=snrs)
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
            if j == 0:
                print(f'are equal: {temp==self.wav}')
                j=j+1
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

    def generate_specific_combinations(self, n_components):
        all_combinations = [np.ones(n_components, dtype=int).tolist()]  # Start with all-ones array
        max_zeros = math.ceil(n_components * 0.30)  # 30% redondeado hacia arriba
        
        for num_zeros in range(1, max_zeros + 1):
            zero_positions = list(combinations(range(n_components), num_zeros))
            
            for positions in zero_positions:
                if len(all_combinations) >= self.num_samples:  
                    return all_combinations
                
                arr = np.ones(n_components, dtype=int)
                arr[list(positions)] = 0
                all_combinations.append(arr.tolist())
        
        return all_combinations

    