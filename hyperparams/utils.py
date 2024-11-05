import pandas as pd
import numpy as np
import logging
from collections import defaultdict
import os
import json
from tqdm import tqdm
import torch 
import ast 

def process_audioset_csv(file_path):
    with open(file_path, 'r') as f:
        next(f)
        lines = f.readlines()
    
    data = []
    for line in lines:
        # Split by the first three commas
        parts = line.strip().split(',', 3)
        if len(parts) == 4:
            ytid = parts[0].strip(' "')
            start = float(parts[1].strip())
            end = float(parts[2].strip())
            # Clean up labels
            labels = parts[3].strip(' "')
            # Split labels and clean them
            labels_list = [label.strip(' "') for label in labels.split(',')]
            data.append([ytid, start, end, labels_list])
    
    df = pd.DataFrame(data, columns=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'])
    
    return df


def get_label_name(label_id, ontology_data):
    labels = []
    for entry in ontology_data:
        if label_id == entry['id']:
            labels.append(entry['name']) 
        if label_id in entry.get('child_ids', []):
            labels.append(entry['name']) 
    return labels


def process_audio_segments(df, ontology_data):
    # Create a new column for storing the label names
    df['name_positive_label'] = df['positive_labels'].apply(
        lambda x: [get_label_name(label_id, ontology_data) for label_id in x if label_id]
    )
    return df

def process_tsv(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df['base_segment_id'] = df['segment_id'].apply(lambda x: x.rsplit('_', 1)[0])
    return df

def open_json(file_path):
    """
    Open and read a JSON file
    """
    with open(file_path, 'r') as f:
        return json.load(f)['real_scores'][0]


def process_predictions_and_labels(base_path, df, model):
    results_dict = {}
    
    for folder in tqdm(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        
        # Skip if it's the 'labels' folder or not a directory
        if folder == 'labels' or not os.path.isdir(folder_path):
            continue
            
        try:
            # Get labels for this folder
            folder_data = df[df['base_segment_id'] == folder]
            if folder_data.empty:
                logging.warning(f"No labels found for folder: {folder}")
                continue
                
            labels = [item for sublist in folder_data['name_positive_label'].iloc[0] 
                     for item in sublist]
            print(labels)
            # Get predictions
            predictions_path = os.path.join(folder_path, 'predictions.json')
            predictions = open_json(predictions_path)
            
            # Track processed label IDs
            processed_ids = set()
            
            # Process positive labels
            for label in labels:
                try:
                    label_id = model.config.label2id[label]
                    processed_ids.add(label_id)
                    
                    key = f'{label}_c'
                    if key not in results_dict:
                        results_dict[key] = []
                    results_dict[key].append(predictions[label_id])
                except KeyError:
                    logging.warning(f"Unknown label: {label} in folder {folder}")
                    continue
            
            # Process remaining labels as negative cases
            for i in range(527):  # Consider making this number configurable
                if i not in processed_ids:
                    try:
                        label = model.config.id2label[i]
                        key = f'{label}_nc'
                        if key not in results_dict:
                            results_dict[key] = []
                        results_dict[key].append(predictions[i])
                    except KeyError:
                        logging.warning(f"Unknown label ID: {i}")
                        continue
                        
        except Exception as e:
            logging.error(f"Error processing folder {folder}: {str(e)}")
            continue
    
    return results_dict


def predict_fn(wav_array, model, feature_extractor):
    if not isinstance(wav_array, list):
        wav_array = [wav_array]
    
    inputs_list = [feature_extractor(audio, sampling_rate=16000, return_tensors="pt") for audio in wav_array]
    
    # Combine the processed features
    inputs = { 
        k: torch.cat([inp[k] for inp in inputs_list]).to('cuda')
        for k in inputs_list[0].keys()
    }
    with torch.no_grad():
        logits = model(**inputs).logits
        
    return logits.cpu().tolist() 

def calculate_confusion_matrix(base_path, df, model, threshold=-0.5):
    """
    Calculate confusion matrix metrics for each class in AudioSet predictions.
    
    Args:
        base_path (str): Path to the root directory containing prediction folders
        df (pd.DataFrame): DataFrame containing ground truth labels
        threshold (float): Threshold for positive predictions (default: -0.5)
    
    Returns:
        dict: Dictionary containing true_positives, false_positives, true_negatives, false_negatives for each class
    """
    metrics = {
        'true_positives': defaultdict(int),
        'false_positives': defaultdict(int),
        'true_negatives': defaultdict(int),
        'false_negatives': defaultdict(int)
    }
    
    # Process each folder
    for folder in tqdm(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        
        # Skip if it's the 'labels' folder or not a directory
        if folder == 'labels' or not os.path.isdir(folder_path):
            continue
            
        try:
            # Get ground truth labels for this folder
            folder_data = df[df['base_segment_id'] == folder]
            if folder_data.empty:
                logging.warning(f"No labels found for folder: {folder}")
                continue
                
            true_labels = ast.literal_eval(folder_data['name_positive_label'].iloc[0])
            true_labels = [item for sublist in true_labels for item in sublist]

            print("Flattened:", true_labels)
            
            
            # Get predictions
            predictions_path = os.path.join(folder_path, 'predictions.json')
            predictions = open_json(predictions_path)
            
            if predictions is None:
                continue
                
            # Process each class
            for label, idx in model.config.label2id.items():
                # Get prediction for this class
                pred_positive = predictions[idx] > threshold
                # Check if this label is in true_labels
                true_positive = label in true_labels
                
                if pred_positive and true_positive:
                    metrics['true_positives'][idx] += 1
                elif pred_positive and not true_positive:
                    metrics['false_positives'][idx] += 1
                elif not pred_positive and true_positive:
                    metrics['false_negatives'][idx] += 1
                else:  # not pred_positive and not true_positive
                    metrics['true_negatives'][idx] += 1
                    
        except Exception as e:
            logging.error(f"Error processing folder {folder}: {e}")
            continue
    
    return metrics