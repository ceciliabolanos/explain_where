"""Process the csv and also get the labels from level 0, 1 and 2. If a label is from level 2,
    we also get the father label. We also calculate the duration of each label. We save the results in a new csv file. """

import pandas as pd
import json 
from utils import get_name_id_mappings
import numpy as np
from transformers import ASTForAudioClassification
import networkx as nx
from typing import List, Dict, Tuple, Set


def get_label_name(label_id, ontology_data):
    for entry in ontology_data:
        if label_id == entry['id']:
            return entry['name']

def get_father_label_name(label_id, mappings):
    return mappings.get(label_id, [np.nan, np.nan, np.nan])[2]


def get_all_ancestors(label_id: str, mappings: Dict[str, Tuple[int, str, str]], 
                     ontology_data: List[dict]) -> Tuple[List[str], List[int]]:
    """
    Get all ancestor labels and their corresponding IDs for a given label ID
    
    Args:
        label_id (str): The ID of the label
        mappings (dict): The hierarchy mappings from get_name_id_mappings
        ontology_data (list): The original ontology data
    
    Returns:
        tuple: (List of ancestor names, List of ancestor IDs)
    """
    # Create graph from ontology
    G = nx.DiGraph()
    for item in ontology_data:
        if 'child_ids' in item:
            for child_id in item['child_ids']:
                G.add_edge(item['id'], child_id)
    
    try:
        # Get all ancestors using networkx
        ancestor_ids = list(nx.ancestors(G, label_id))
        # Get names and model IDs for all ancestors
        ancestor_names = [get_label_name(ancestor_id, ontology_data) for ancestor_id in ancestor_ids]
        ancestor_model_ids = [model.config.label2id.get(name, -1) for name in ancestor_names if name is not None]
        return ancestor_names, ancestor_model_ids
    except:
        return [], []
    
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
ancestor_results = df_segmented['label'].apply(
    lambda x: get_all_ancestors(x, mappings, ontology_data)
)

df_segmented['other_labels'] = ancestor_results.apply(lambda x: x[0])
df_segmented['other_labels_ids'] = ancestor_results.apply(lambda x: x[1])

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


