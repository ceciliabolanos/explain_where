import json
import numpy as np
from tqdm import tqdm
import re
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
        
        if intersection_end > intersection_start:
            intersection_duration = intersection_end - intersection_start
            if intersection_duration >= 0.06:
                return 1 # capaz intesection deberia ser el resultado
    return 0

def generate_sequence(length):
    return [i * 0.1 for i in range(length)]

def create_segmentation_vector(times_gt, times, granularidad_ms): 
    real_vector = np.zeros(len(times))

    for i in range(len(real_vector)):
        real_vector[i] = time_in_segmentation(times[i], times_gt, granularidad_ms)
    
    return real_vector


def process_audio_file(data, method):
    segment_length = data["metadata"]['segment_length']
    filename = data["metadata"]['filename']
    overlap = 0 
    granularidad_ms = segment_length - overlap
    times_gt = data['metadata']["true_markers"]
    
    # Get importance scores
    if method == 'tree_importance':
        values = data['importance_scores']['random_forest_tree_importance']['values']
    elif method == 'shap':
        values = data['importance_scores']['random_forest_shap_importance']['values']
    elif method == 'naive':
        values = data['importance_scores']['naive']['values']
    elif method == 'linear_regression':
        values = data['importance_scores']['linear_regression']['values']['coefficients']
    elif method == 'kernel_shap':
        values = data['importance_scores']['kernel_shap']['values']['coefficients']
    times = generate_sequence(len(values))
    
    times_segmentation = create_segmentation_vector(times_gt, times, granularidad_ms)

    sorted_indices = np.argsort(values)[::-1]  # Sort indices in descending order
    ranking_scores = np.zeros_like(values)
    ranking_scores[sorted_indices] = np.linspace(1, 0, len(values))
    
    # Calculate ROC curve points
    fpr, tpr, thresholds = roc_curve(np.array(times_segmentation), ranking_scores)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    if sum(times_segmentation) == 0:
        print(filename)
    
    # Calculate scores
    order_segmentation_values = {
        'real_order': times_segmentation, # Original order of segmentation values (0s and 1s)
        'model_order': values, # Segmentation values sorted by importance
        'roc_auc': roc_auc,
        'k': sum(times_segmentation), # Total number of segments in ground truth
    }

    return {
        'filename': filename,
        'event_label': data['metadata']['id_explained'],
        'actual_score': data['metadata']['true_score'],
        **order_segmentation_values,
        'true_markers': times_gt,
    }

def get(method: str, mask_percentage, window_size, mask_type, function, base_path: str, dataset):
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, f'explanations_{dataset}'))):
        pattern = re.compile(rf'ft_.*_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}\.json$')
        json_files = [f for f in files if pattern.match(f)]
        
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            with open(file_path, 'r') as file:
                data = json.load(file)

            result = process_audio_file(data, method)
            results.append(result)
        
    output_dir = os.path.join(f'evaluations/{dataset}/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'order_{method}_p{mask_percentage}_f{function}_m{mask_type}.tsv'), sep='\t', index=False)

def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/ec2-user/results1',
                      help='Base path for AudioSet experiments')
    args = parser.parse_args()

    # for function in ['euclidean', 'cosine', 'dtw']:
    #     for mask_type in ['zeros', 'stat', 'noise']:
    #         for mask_percentage in [0.1, 0.2, 0.3, 0.4]:
    #             for window_size in [1, 2, 3, 4, 5]:
    #                 for method in ['tree_importance', 'shap', 'naive', 'linear_regression', 'greedy', 'kernel_shap']:
    #                     get(method, mask_percentage, window_size, mask_type, function, args.base_path, 'audioset')
    
    # cough 
    for function in ['euclidean', 'cosine', 'dtw']:
        for mask_type in ['zeros', 'stat', 'noise']:
            for mask_percentage in [0.2, 0.3, 0.4]:
                for window_size in [1, 3, 5]:
                    for method in ['tree_importance', 'shap', 'naive', 'linear_regression', 'kernel_shap']:
                        get(method, mask_percentage, window_size, mask_type, function, args.base_path, 'cough')

if __name__ == '__main__':
    main()

    