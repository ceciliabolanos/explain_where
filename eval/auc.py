import json
import numpy as np
from tqdm import tqdm
from utils import get_patterns
import os
import pandas as pd
import argparse
from sklearn.metrics import roc_curve, auc


def time_in_segmentation(max_index, list_of_tuples, granularidad_ms):
    interval_start = max_index
    interval_end = max_index + granularidad_ms/1000
    
    for tuple_start, tuple_end in list_of_tuples:
        # Calculate intersection
        intersection_start = max(interval_start, tuple_start)
        intersection_end = min(interval_end, tuple_end)
        
        # If there is an intersection, check its duration
        if intersection_end > intersection_start:
            intersection_duration = intersection_end - intersection_start
            if intersection_duration >= 0.1:
                return 1 # capaz intesection deberia ser el resultado
    return 0


def generate_sequence(length):
    return [i * 0.250 for i in range(length)]


def create_segmentation_vector(times_gt, times, granularidad_ms): 
    # Initialize result vectors
    real_vector = np.zeros(len(times))
    
    for i in range(len(real_vector)):
        real_vector[i] = time_in_segmentation(times[i], times_gt, granularidad_ms)
    
    return real_vector


def process_audio_file(data, method):
    id_explained = data['metadata']['id_explained']
    segment_length = data["metadata"]['segment_length']
    filename = data["metadata"]['filename']
    overlap = data["metadata"]['overlap']
    granularidad_ms = segment_length - overlap
    times_gt = data['metadata']["true_markers"]
    
    # Get importance scores
    if method == 'tree_importance':
        values = data['importance_scores']['random_forest'][method]['values']
    elif method == 'shap':
        values = data['importance_scores']['random_forest'][method]['values']
    elif method == 'naive':
        values = data['importance_scores'][method]['values']
    elif method == 'linear_regression':
        values = data['importance_scores'][method]['masked']['values']['coefficients']
   
    times = generate_sequence(len(values))
    
    times_segmentation = create_segmentation_vector(times_gt, times, granularidad_ms)

    sorted_indices = np.argsort(values)[::-1]  # Sort indices in descending order
    ranking_scores = np.zeros_like(values)
    ranking_scores[sorted_indices] = np.linspace(1, 0, len(values))
    
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(np.array(times_segmentation), ranking_scores)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    # Calculate scores
    order_segmentation_values = {
        'real_order': times_segmentation, # Original order of segmentation values (0s and 1s)
        'model_order': values, # Segmentation values sorted by importance
        'roc_auc': roc_auc,
        'k': sum(times_segmentation), # Total number of segments in ground truth
    }

    return {
        'filename': filename,
        'event_label': data['metadata']['label_explained'],
        'actual_score': data['metadata']['true_score'],
        **order_segmentation_values
    }

def get(method: str, base_path: str):
    patterns = get_patterns()
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, 'audioset_audios_eval'))):
        json_files = [f for pattern in patterns for f in files if pattern.match(f)]
        
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            with open(file_path, 'r') as file:
                data = json.load(file)

            result = process_audio_file(data, method)
            results.append(result)
        
    # Save results
    output_dir = os.path.join(base_path, 'audioset_evaluation/auc')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'order_{method}.tsv'), sep='\t', index=False)



def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/cbolanos/experiments',
                      help='Base path for AudioSet experiments')

    args = parser.parse_args()
    for method in ['tree_importance', 'shap', 'naive', 'linear_regression']:
        get(method, args.base_path)


if __name__ == '__main__':
    main()

    
    