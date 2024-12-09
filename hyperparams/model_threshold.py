
import json
from utils import open_json
import os
import pandas as pd
from tqdm import tqdm
import logging 
from collections import defaultdict


base_path = '/home/cbolanos/experiments/audioset_audios_eval/'
df = pd.read_csv('/home/cbolanos/experiments/audioset/labels/labels_segments.csv')
json_name = 'predictions_ast.json'

############# Process once we have the score to get the distribution #############
results_dict = {}

for folder in tqdm(os.listdir(base_path)):
    folder_path = os.path.join(base_path, folder)
    
    try:
        # Get labels for this folder
        folder_data = df[df['base_segment_id'] == folder]
        true_ids = folder_data['father_labels_ids'].tolist()
        
        # Get predictions
        predictions_path = os.path.join(folder_path, json_name)
        predictions = open_json(predictions_path)
        
        # Track processed label IDs
        processed_ids = set()
        
        # Process positive labels
        for id in true_ids:
            processed_ids.add(id)
            if id in predictions['positive_indices']:
                key = f'{id}_tp'
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].append(predictions['real_scores'][0][id])
            else:
                key = f'{id}_fn'
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].append(predictions['real_scores'][0][id])
 
        
        for id in predictions['positive_indices']:
            if id not in processed_ids:
                key = f'{id}_fp'
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].append(predictions['real_scores'][0][id])

                    
    except Exception as e:
        logging.error(f"Error processing folder {folder}: {str(e)}")
        continue
    

output_path = f'/home/cbolanos/experiments/audioset/labels/distributions.json'
with open(output_path, 'w') as f:
    json.dump(results_dict, f, indent=2)


############ Process and get the metrics #############

# metrics = {
#     'true_positives': defaultdict(int),
#     'false_positives': defaultdict(int),
#     'true_negatives': defaultdict(int),
#     'false_negatives': defaultdict(int)
# }

# for folder in tqdm(os.listdir(base_path)):
#     folder_path = os.path.join(base_path, folder)
    
#     # Skip if it's the 'labels' folder or not a directory
#     if folder == 'labels' or not os.path.isdir(folder_path):
#         continue
        
#     try:
#         # Get ground truth labels for this folder
#         folder_data = df[df['base_segment_id'] == folder]
        
#         if folder_data.empty:
#             logging.warning(f"No labels found for folder: {folder}")
#             continue
#         true_labels = (folder_data['name_positive_label'].iloc[0])
#         true_labels = [item for sublist in true_labels for item in sublist]
                
#         # Get predictions
#         predictions_path = os.path.join(folder_path, 'predictions.json')
#         predictions = open_json(predictions_path)
        
#         if predictions is None:
#             continue
            
#         # Process each class
#         for label, idx in model.config.label2id.items():
#             # Get prediction for this class
#             pred_positive = predictions[idx] > threshold
#             # Check if this label is in true_labels
#             true_positive = label in true_labels
            
#             if pred_positive and true_positive:
#                 metrics['true_positives'][idx] += 1
#             elif pred_positive and not true_positive:
#                 metrics['false_positives'][idx] += 1
#             elif not pred_positive and true_positive:
#                 metrics['false_negatives'][idx] += 1
#             else:  # not pred_positive and not true_positive
#                 metrics['true_negatives'][idx] += 1
                
#     except Exception as e:
#         logging.error(f"Error processing folder {folder}: {e}")
#         continue

# output_path = f'/home/cbolanos/experiments/audioset/labels/metrics.json'
# with open(output_path, 'w') as f:
#     json.dump(metrics, f, indent=2)
