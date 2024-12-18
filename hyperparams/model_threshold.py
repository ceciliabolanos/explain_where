
import json
from utils import open_json
import os
import pandas as pd
from tqdm import tqdm
import logging 
import ast

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
        other_true_ids = []
        for id_list_str in folder_data['other_labels_ids']:
            id_list = ast.literal_eval(id_list_str)  # Convert string to list
            other_true_ids.extend([x for x in id_list if x != -1])  # Optional: filter out -1s

        # Get predictions
        predictions_path = os.path.join(folder_path, json_name)
        predictions = open_json(predictions_path)
        
        # Track processed label IDs
        processed_ids = set()
        
        # Process positive labels
        for id in true_ids:
            processed_ids.add(id)
            key = f'{id}_true'
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append(predictions['real_scores'][0][id])
        
        for id in other_true_ids:
            if id not in processed_ids and id != -1:
                processed_ids.add(id)
                key = f'{id}_true'
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].append(predictions['real_scores'][0][id])

        for id in range(527):
            if id not in processed_ids:
                key = f'{id}_false'
                if key not in results_dict:
                    results_dict[key] = []
                results_dict[key].append(predictions['real_scores'][0][id])

    except Exception as e:
        logging.error(f"Error processing folder {folder}: {str(e)}")
        continue
    

output_path = f'/home/cbolanos/experiments/audioset/labels/distributions.json'
with open(output_path, 'w') as f:
    json.dump(results_dict, f, indent=2)

