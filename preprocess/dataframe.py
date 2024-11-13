
import pandas as pd
import json 
from utils import get_name_id_mappings
import numpy as np

def get_label_name(label_id, ontology_data):
    for entry in ontology_data:
        if label_id == entry['id']:
            return entry['name']

def get_father_label_name(label_id, mappings):
    return mappings.get(label_id, [np.nan, np.nan, np.nan])[2]

############# Preprocess to get the audio files we are going to run and the labels we have segmented for each one. #############

df_segmented = pd.read_csv('/home/cbolanos/experiments/audioset/labels/audioset_train_strong.tsv', sep='\t')
df_segmented['base_segment_id'] = df_segmented['segment_id'].apply(lambda x: x.rsplit('_', 1)[0])    

with open('/home/cbolanos/experiments/audioset/labels/ontology.json', 'r') as file:
     ontology_data = json.load(file)

df_segmented['positive_labels'] = df_segmented['label'].apply(
        lambda x: get_label_name(x, ontology_data)
    )

mappings = get_name_id_mappings(ontology_data)

df_segmented['father_labels'] = df_segmented['label'].apply(
        lambda x: get_father_label_name(x, mappings)
    )

df_segmented = df_segmented[df_segmented['father_labels'].notna()]
df_segmented['label_duration'] = (df_segmented['end_time_seconds'] - df_segmented['start_time_seconds']) * 1000
df_segmented.to_csv('/home/cbolanos/experiments/audioset/labels/labels_segments.csv')


