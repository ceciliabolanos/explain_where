"""Preprocess the files to get the predictions and ground truth for the PSDS evaluation."""
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from utils import process_importance_values, get_patterns
import pandas as pd
import soundfile as sf
from scipy.signal import resample


def get(method: str, threshold: float, base_path: str):
    """
    Process AudioSet data and generate evaluation files.
    
    Args:
        score (float): Score threshold for pattern matching
        method (str): Analysis method ('random_forest', 'naive', or 'lime')
        threshold (float): Threshold for importance values
        base_path (str): Base path for AudioSet experiments
    """
    patterns = get_patterns()

    data_gt = {'filename': [], 'onset': [], 'offset': [], 'event_label': [], 'true_score': []}
    data_metadata = {'filename': [], 'duration': []}
    data_threshold = {'filename': [], 'onset': [], 'offset': [], 'event_label': [],'true_score': []}
    
    for root, dirs, files in tqdm(os.walk(os.path.join(base_path, 'audioset_audios_eval'))):
        json_files = []
        for pattern in patterns:
            json_files.extend([f for f in files if pattern.match(f)])
            
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            class_explained = data['metadata']['label_explained']
            segment_length = data["metadata"]['segment_length']
            filename = data["metadata"]['filename']
            overlap = data["metadata"]['overlap']
            granularidad_ms = segment_length - overlap
            times_gt = data['metadata']["true_markers"]

            wav_data, sample_rate = sf.read(f'/mnt/shared/alpha/hdd6T/Datasets/audioset_eval_wav/{filename}.wav')
            if sample_rate != 16000:
                wav_data = resample(wav_data, int(len(wav_data) * 16000 / sample_rate))
         
            if len(wav_data.shape) > 1 and wav_data.shape[1] == 2:
                wav_data = wav_data.mean(axis=1)

            duration = len(wav_data)/16000
            # Get importance scores
            
            if method == 'random_forest':
                values = data['importance_scores'][method]['masked']['values']
            elif method == 'naive':
                values = data['importance_scores'][method]['values']
            elif method == 'lime':
                values = data['importance_scores'][method]['masked']['values']['coefficients']
            elif method == 'yamnet':
                predictions_path = f'{base_path}/audioset_audios_eval/{filename}/predictions_yamnet.json'
                with open(predictions_path, 'r') as f:
                    scores_yamnet = np.array(json.load(f)['real_scores']) 
                with open('/home/cbolanos/experiments/audioset/labels/labels_yamnet.json', 'r') as f:
                    class_names = json.load(f)['label']
                index = class_names.index(class_explained)
                values = scores_yamnet[:, index]
                times = np.linspace(0, duration, len(values))

            if method == 'yamnet':
                j = 1
                timestep = duration/len(values)
                importance_values, times = values, times
            else: 
                j = 250
                timestep = 0.25
                importance_values, times = process_importance_values(values, segment_size=segment_length, step_size=granularidad_ms)
        
            importancias_normalizadas = importance_values.copy()
            if importancias_normalizadas.max() != importancias_normalizadas.min():  # Check if all values aren't the same
                importancias_normalizadas = (importancias_normalizadas - importancias_normalizadas.min()) / \
                                        (importancias_normalizadas.max() - importancias_normalizadas.min())
            else:
                importancias_normalizadas[:] = 0  # If all values are the same, set everything to 0
            
            current_segment = None
            # For threshold predictions
            for i in range(0, len(importancias_normalizadas), j):
                importancia = importancias_normalizadas[i]
                time = times[i]  # assuming times array matches importancias_normalizadas
                
                if importancia >= threshold:
                    if current_segment is None:
                        first_time = time
                        ocurrencies = 0
                    ocurrencies =+ timestep
                    current_segment = True
                
                elif importancia < threshold and current_segment is not None:
                    current_segment = None
                    data_threshold['filename'].append(filename)
                    data_threshold['event_label'].append(class_explained)
                    data_threshold['onset'].append(first_time)
                    data_threshold['offset'].append(first_time + ocurrencies)
                    data_threshold['true_score'].append(data['metadata']["true_score"])

            # For ground truth
            for time_tuple in times_gt:
                data_gt['filename'].append(filename)
                data_gt['event_label'].append(class_explained)
                data_gt['onset'].append(time_tuple[0])
                data_gt['offset'].append(time_tuple[1])
                data_gt['true_score'].append(data['metadata']["true_score"])

            # For metadata
            data_metadata['filename'].append(filename)
            data_metadata['duration'].append(len(wav_data) / 16000)


    # Save results as TSV files
    output_dir = os.path.join(base_path, 'audioset_evaluation/psds')
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    pred_df = pd.DataFrame(data_threshold)
    pred_file = os.path.join(output_dir, f'predictions_{method}/{threshold}.tsv')
    os.makedirs(os.path.dirname(pred_file), exist_ok=True)
    pred_df.to_csv(pred_file, sep='\t', index=False)

    # Save ground truth
    gt_df = pd.DataFrame(data_gt)
    gt_file = os.path.join(output_dir, 'ground_truth.tsv')
    gt_df.to_csv(gt_file, sep='\t', index=False)

    # Save metadata
    meta_df = pd.DataFrame(data_metadata)
    meta_file = os.path.join(output_dir, 'metadata.tsv')
    meta_df.to_csv(meta_file, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/cbolanos/experiments',
                      help='Base path for AudioSet experiments')

    args = parser.parse_args()
    thresholds = np.arange(0.03, 1.00, 0.02).round(3)
    for method in ['random_forest', 'naive', 'lime', 'yamnet']:
        for t in thresholds:
            get(method, t, args.base_path)

if __name__ == "__main__":
    main()