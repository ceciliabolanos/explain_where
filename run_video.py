import os
from tqdm import tqdm
import logging
import pandas as pd
from utils import open_json
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from all_methods import run_all_methods, generate_data

FOLDER = '3B0WbwAkByI'
BASE_PATH = '/home/cbolanos/experiments/audioset_audios_eval/'
LABEL = 'Speech'
df = pd.read_csv('/home/cbolanos/experiments/audioset/labels/labels_segments.csv')

model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = model.to('cpu')

json_name = 'predictions_ast.json'

folder_path = os.path.join(BASE_PATH, FOLDER)

# Get ground truth labels for this folder
folder_data = df[df['base_segment_id'] == FOLDER]

# Get predictions
predictions_path = os.path.join(folder_path, json_name)
predictions = open_json(predictions_path)
    
true_ids = folder_data['father_labels_ids'].tolist()     
ids_intersection = list(set(predictions['positive_indices']) & set(true_ids))

# Process each label
for id in ids_intersection:
    label = model.config.id2label[id]
    if label == LABEL:
        print(label)
        mask = (folder_data['father_labels'] == label)
        time_tuples = list(zip(
                folder_data[mask]['start_time_seconds'],
                folder_data[mask]['end_time_seconds']
        ))

        results = run_all_methods(
            filename=FOLDER,
            id_to_explain=id,
            label_to_explain=label,
            markers=time_tuples,
            segment_length=500,
            overlap=250,
            true_score=predictions['real_scores'][0][id],
            num_samples=4500,
            generate_video=True
        )
        

                
