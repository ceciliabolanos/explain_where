import json
import numpy as np
from tqdm import tqdm
import re
import os
import pandas as pd
import argparse
from sklearn.metrics import roc_curve, auc

def time_in_segmentation(max_index, list_of_tuples, granularidad_ms, interesection_duration_threshold):
    interval_start = max_index
    interval_end = max_index + granularidad_ms/1000
    
    for tuple_start, tuple_end in list_of_tuples:

        intersection_start = max(interval_start, tuple_start)
        intersection_end = min(interval_end, tuple_end)
        
        if intersection_end > intersection_start:
            intersection_duration = intersection_end - intersection_start
            if intersection_duration > interesection_duration_threshold:
                return 1
    return 0

def generate_sequence(length):
    return [i * 0.1 for i in range(length)]

def create_segmentation_vector(times_gt, times, granularidad_ms, intersection): 
    real_vector = np.zeros(len(times))

    for i in range(len(real_vector)):
        real_vector[i] = time_in_segmentation(times[i], times_gt, granularidad_ms, intersection)
    
    return real_vector

def compute_importance_score(importance_values, gt_mask):
    importance_values = np.array(importance_values)
    gt_mask = np.array(gt_mask)
    
    # Encontrar índices donde gt_mask es 1
    gt_indices = np.where(gt_mask == 1)[0]
    non_gt_indices = np.where(gt_mask == 0)[0]
    
    # Encontrar regiones continuas en GT
    from itertools import groupby
    from operator import itemgetter
    
    def get_regions(indices):
        regions = []
        current_region = []
        for i in range(len(indices)):
            if i == 0 or indices[i] == indices[i - 1] + 1:
                current_region.append(indices[i])
            else:
                regions.append(current_region)
                current_region = [indices[i]]
        if current_region:
            regions.append(current_region)
        return regions
    
    gt_regions = get_regions(gt_indices)

    # Obtener el máximo de cada región en GT y luego su mínimo
    max_in_regions = [max(importance_values[region]) for region in gt_regions]
    min_of_max_gt = min(max_in_regions) if max_in_regions else float('-inf')
    
    # Obtener el máximo fuera de GT
    max_outside_gt = max(importance_values[non_gt_indices]) if len(non_gt_indices) > 0 else float('-inf')
    if max_outside_gt < 0:
        max_outside_gt = 0
    # Rango de importancias
    global_max = max(importance_values)
    range_importance = global_max if global_max != 0 else 1  # Evitar división por cero
    
    # Calcular métrica
    score = (min_of_max_gt - max_outside_gt) / range_importance
    
    return score

def process_audio_file(data, method, intersection):
    granularidad_ms = data["metadata"]['segment_length']
    filename = data["metadata"]['filename']
    times_gt = data['metadata']["true_markers"]
    
    # Get importance scores
    if method == 'tree_importance':
        values = data['importance_scores']['random_forest_tree_importance']['values']
    elif method == 'linear_regression':
        values = data['importance_scores']['linear_regression']['values']['coefficients']
    elif method == 'kernel_shap':
        values = data['importance_scores']['kernel_shap']['values']['coefficients']
    elif method == 'linear_regression_noreg_noweights':
        values = data['importance_scores']['linear_regression_nocon']['values']['coefficients']
    elif method == 'kernel_shap_sumcons':
        values = data['importance_scores']['importances_kernelshap_analyzer_1constraint']['values']['coefficients']
    times = generate_sequence(len(values))
    
    times_segmentation = create_segmentation_vector(times_gt, times, granularidad_ms, intersection)

    if sum(times_segmentation) == 0:
        print(filename)

    leo_metric = compute_importance_score(values, times_segmentation)
    
    order_segmentation_values = {
        'real_order': times_segmentation, 
        'model_order': values,
        'leo_metric': leo_metric,
        'k': sum(times_segmentation), 
    }

    return {
        'filename': filename,
        'event_label': data['metadata']['id_explained'],
        'actual_score': data['metadata']['true_score'],
        **order_segmentation_values,
        'true_markers': times_gt,
    }

def get(method: str, mask_percentage, window_size, mask_type, function, base_path: str, dataset, intersection):
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, f'explanations_{dataset}'))):
        pattern = re.compile(rf'ft1_.*_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}\.json$')
        json_files = [f for f in files if pattern.match(f)]
        
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}. Retrying {file_path}")
           
            result = process_audio_file(data, method, intersection)
            results.append(result)
        
    output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'leo_metric_{method}_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}_{intersection}.tsv'), sep='\t', index=False)


def get_with_name(method: str, name, base_path: str, dataset, intersection):
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, f'explanations_{dataset}'))):
        pattern = re.compile(rf'ft_.*_{name}\.json$')
        json_files = [f for f in files if pattern.match(f)]
        
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}. Retrying {file_path}")
           
            result = process_audio_file(data, method, intersection)
            results.append(result)
        
    output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'leo_metric_{method}_{name}_{intersection}.tsv'), sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/ec2-user/results1',
                      help='Base path for AudioSet experiments')
    args = parser.parse_args()

    # Select the dataset to run

    names = ["zeros", "noise", "stat", "all"] 
    dataset = 'kws'
    for function in ['euclidean']:
        for mask_type in ['zeros', 'stat', 'noise']:
            for mask_percentage in [0.2, 0.3, 0.4]:
                for window_size in [1, 3, 5]:
                    for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
                        get(method, mask_percentage, window_size, mask_type, function, args.base_path, dataset, 0)
    for name in names:
        for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
            get_with_name(method, name, args.base_path, dataset, 0)
  
    # names = ["0.2-1", "0.2-3", "0.2-5",
    #         "0.3-1", "0.3-3", "0.3-5",
    #         "0.4-1","0.4-3","0.4-5", 
    #          "zeros", "noise", "stat", "all"]

if __name__ == '__main__':
    main()

    