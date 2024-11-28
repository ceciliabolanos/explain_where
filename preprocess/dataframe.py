"""Process the csv and also get the labels from level 0, 1 and 2. If a label is from level 2,
    we also get the father label. We also calculate the duration of each label. We save the results in a new csv file. """

import pandas as pd
import json 
from utils import get_name_id_mappings
import numpy as np
from transformers import ASTForAudioClassification

def get_label_name(label_id, ontology_data):
    for entry in ontology_data:
        if label_id == entry['id']:
            return entry['name']

def get_father_label_name(label_id, mappings):
    return mappings.get(label_id, [np.nan, np.nan, np.nan])[2]

############# Preprocess to get the audio files we are going to run and the labels we have segmented for each one. #############

df_segmented = pd.read_csv('/home/cbolanos/experiments/audioset/labels/audioset_eval_strong.tsv', sep='\t')
df_segmented['base_segment_id'] = df_segmented['segment_id'].apply(lambda x: x.rsplit('_', 1)[0])

model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

with open('/home/cbolanos/experiments/audioset/labels/ontology.json', 'r') as file:
     ontology_data = json.load(file)

df_segmented['positive_labels'] = df_segmented['label'].apply(
        lambda x: get_label_name(x, ontology_data)
    )

mappings = get_name_id_mappings(ontology_data)

df_segmented['father_labels'] = df_segmented['label'].apply(
        lambda x: get_father_label_name(x, mappings)
    )
df_segmented['father_labels_ids'] = df_segmented['father_labels'].apply(
    lambda x: model.config.label2id.get(x, -1)
).astype('Int64')

df_segmented = df_segmented[df_segmented['father_labels_ids'] != -1]
df_segmented = df_segmented[df_segmented['father_labels'].notna()]
df_segmented['label_duration'] = (df_segmented['end_time_seconds'] - df_segmented['start_time_seconds']) * 1000
df_segmented.to_csv('/home/cbolanos/experiments/audioset/labels/labels_segments.csv')


