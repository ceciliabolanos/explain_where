import os
import json
import numpy as np
from tqdm import tqdm
import logging
import pandas as pd
from utils import open_json
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from all_methods import run_all_methods, generate_data
import ast 


# 1) Select the audios (instances) where we should evaluate. The ones which where correctly predicted by the model and also that we have the gt.

base_path = '/home/cbolanos/experiments/audioset/'
df = pd.read_csv('/home/cbolanos/experiments/audioset/labels/labels_segments.csv')

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = model.to('cuda')

threshold = -0.5 # this threshold has been selected in /hyperparams

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

        generate_data(folder, 
                  model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                  segment_length=500,
                  overlap=400,
                  num_samples=1000)
        
        # Process each class
        for label, idx in model.config.label2id.items():
            pred_positive = predictions[idx] > threshold
            true_positive = label in true_labels

            if pred_positive and true_positive:
                results = run_all_methods(
                    filename=folder,
                    id_to_explain=idx,
                    label_to_explain=label,
                    segment_length=500,
                    overlap=400,
                    num_samples=1000
                )

    except Exception as e:
        logging.error(f"Error processing folder {folder}: {str(e)}")
        continue 
                    

# 3) Select the threshold to determine if a segment is important or not
