"""Is the maximum importance value inside the ground truth marker?"""

import os
import json 
import argparse
from utils import process_importance_values
import numpy as np
from tqdm import tqdm
from utils import get_patterns

def is_contained_in_any(interval, list_of_tuples):
    start, end = interval
    for tuple_start, tuple_end in list_of_tuples:
        if tuple_start <= start and end <= tuple_end:
            return 1
    return 0

def max_inside_marker(max_index, list_of_tuples):
    for tuple_start, tuple_end in list_of_tuples :
        if tuple_start <= max_index and max_index <= tuple_end:
            return 1
    return 0
 

def obtener_regiones_detectadas(data, method, segment_length, granularidad_ms, mode='global'):
    if method == 'random_forest':
        values = data['importance_scores'][method]['masked']['values']
    elif method == 'naive':
        values = data['importance_scores'][method]['values']
    elif method == 'lime':
        values = data['importance_scores'][method]['masked']['values']['coefficients']

    importance_values, times = process_importance_values(values, segment_size=segment_length, step_size=granularidad_ms)
    
    if mode == 'global':
        max_index = np.argmax(importance_values)
        return (times[max_index], times[max_index] + granularidad_ms/1000)
    elif mode == 'marker_max':
        max_index = np.argmax(importance_values)
        return times[max_index]
    else:
        raise ValueError("Invalid mode. Choose 'global' or 'marker_max'")

def evaluate_yanmet(
    path: str,
    method: str,
    yamnet_path: str, 
    mode='global') -> int:
   
    with open(path, 'r') as file:
        data = json.load(file)

    with open(yamnet_path, 'r') as f:
        scores_yamnet = np.array(json.load(f)['real_scores'])
    
    with open('/home/cbolanos/experiments/audioset/labels/labels_yamnet.json', 'r') as f:
        class_names = json.load(f)['label']

    label_to_explain = path.split('_')[-1].replace('.json','')
    index = class_names.index(label_to_explain)

    class_scores = scores_yamnet[:, index]
    max_index =np.argmax(class_scores)
    
    times_gt = data['metadata']["true_markers"]
    detected_regions = (max_index*0.5-0.250, max_index*0.5+0.250)
 
    if mode == 'global':
        return is_contained_in_any(detected_regions, times_gt)
    elif mode == 'marker_max':
        return max_inside_marker(detected_regions[0], times_gt)

def evaluar_deteccion_regiones(
    path: str,
    method: str,
    mode='global') -> int:

    with open(path, 'r') as file:
        data = json.load(file)

    segment_length = data["metadata"]['segment_length']
    overlap =  data["metadata"]['overlap']
    granularidad_ms = segment_length - overlap
    times_gt =   data['metadata']["true_markers"]

    detected_regions = obtener_regiones_detectadas(
        data, 
        method,  
        segment_length=segment_length,
        granularidad_ms=granularidad_ms
    )
 
    if mode == 'global':
        return is_contained_in_any(detected_regions, times_gt)
    elif mode == 'marker_max':
        return max_inside_marker(detected_regions[0], times_gt)

    
def evaluate_all_folders(base_path: str, method: str, score: float, mode: str) -> dict:
    results = {
        "summary": {
            "total_files": 0,
            "successful_assertions": 0,
            "failure_rate": 0.0,
            "method": method,
            "score_threshold": score,
            "base_path": base_path
        },
        "failed_files": [],
        "detailed_results": []
    }
    
    patterns = get_patterns()
    
    for root, dirs, files in tqdm(os.walk(os.path.join(base_path, "audioset_audios_eval"))):
        # Filter for JSON files matching any of the patterns
        json_files = []
        for pattern in patterns:
            json_files.extend([f for f in files if pattern.match(f)])
        
        for json_file in json_files:
            file_path = os.path.join(root, json_file)

            with open(file_path, 'r') as file:
                data = json.load(file)

            if data['metadata']["true_score"] <= score:
                continue

            if method=='yamnet': 
                result = evaluate_yanmet(file_path, method, yamnet_path=os.path.join(root, "predictions_yamnet.json"), mode=mode)
            else:
                result = evaluar_deteccion_regiones(file_path, method, mode)
            # Update counters
            results["summary"]["total_files"] += 1
            results["summary"]["successful_assertions"] += result
            
            # Store detailed result
            detail = {
                "file_path": file_path,
                "passed": bool(result)
            }
            results["detailed_results"].append(detail)
            
            # Track failed paths
            if result == 0:
                results["failed_files"].append(file_path)
    
    # Calculate failure rate
    total = results["summary"]["total_files"]
    if total > 0:
        results["summary"]["failure_rate"] = (
            len(results["failed_files"]) / total * 100
        )
    
    return results

def save_results(results: dict, base_path: str, method: str, score: float, mode):
    # Create results directory if it doesn't exist
    results_dir = os.path.join(base_path, "audioset_evaluation/max/evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filename with timestamp
    filename = f"{method}_{score}_{mode}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Save results to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return filepath
    
def main():
    base_path='/home/cbolanos/experiments'
    for method in ['random_forest', 'naive', 'lime', 'yamnet']:
        for mode in ['global', 'marker_max']:
            for score in [-2, -1, 0, 1, 2]:
                results = evaluate_all_folders(
                    base_path,
                    method,
                    score,
                    mode
                )

                output_file = save_results(
                    results,
                    base_path,
                    method,
                    score,
                    mode
                )
    
                # Print summary to console
                print("\nEvaluation Summary:")
                print(f"Total files processed: {results['summary']['total_files']}")
                print(f"Successful assertions: {results['summary']['successful_assertions']}")
                print(f"Failure rate: {results['summary']['failure_rate']:.2f}%")
                print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()