import os
from tqdm import tqdm
import logging
import pandas as pd
from utils import open_json
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from all_methods import run_all_methods, generate_data
import json

FOLDER = 'uoP8GoweWBI'
threshold = 0.0   

BASE_PATH = '/home/cbolanos/experiments/audioset/'
df = pd.read_csv('/home/cbolanos/experiments/audioset/labels/labels_segments.csv')
data_generate = True
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = model.to('cuda')
json_name = 'predictions_ast.json'

folder_path = os.path.join(BASE_PATH, FOLDER)
    
try:
    # Get ground truth labels for this folder
    folder_data = df[df['base_segment_id'] == FOLDER]
    if folder_data.empty:
        logging.warning(f"No labels found for folder: {FOLDER}")
        
    true_labels = folder_data['father_labels'].tolist()

    print("Flattened:", true_labels)
    
    # Get predictions
    predictions_path = os.path.join(folder_path, json_name)
    predictions = open_json(predictions_path)
    if not data_generate:
        generate_data(FOLDER, 
                    model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                    segment_length=500,
                    overlap=400,
                    num_samples=7000)
    
    for label, idx in model.config.label2id.items():
        pred_positive = predictions[idx] > threshold
        true_positive = label in true_labels

        if pred_positive and true_positive: # Here we might change this and penalize the model if we don't predict the label we want
            print(f'We could predict {label}')    

            mask = (folder_data['father_labels'] == label)

            time_tuples = list(zip(
                folder_data[mask]['start_time_seconds'],
                folder_data[mask]['end_time_seconds']
            ))

            results = run_all_methods(
                filename=FOLDER,
                id_to_explain=idx,
                label_to_explain=label,
                markers=time_tuples,
                segment_length=500,
                overlap=400,
                num_samples=7000, 
                threshold=threshold
            )
        elif true_positive:
            print(f'We could not predict {label}')    

except Exception as e:
    logging.error(f"Error processing folder {FOLDER}: {str(e)}")
 
                
