import pandas as pd
from tqdm import tqdm
from run_explanation import generate_explanation, generate_explanation_from_file
from data.base_generator import MaskingConfig
from data.random_generator import RandomDataGenerator

import json
from multiprocessing import Pool, cpu_count
from itertools import product
import os

LABELS_PATH = '/home/ec2-user/Datasets/Audioset/labels/audioset_eval_train.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 3000
df_chunk = pd.read_csv(LABELS_PATH)

# Hyperparameters
mask_percentages = [0.3, 0.4, 0.2]
window_sizes = [5, 1, 3]
mask_types = ['zeros','stat','noise']

selected_files = pd.read_csv('/home/ec2-user/explain_where/preprocess/files_to_process.csv')

mask_configs = [ 
    {"zeros": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros'], "combinations": 9}},
    {"noise": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "combinations": 9}}
]


for i in tqdm(range(len(selected_files))):
    if i < 2:
        filename = selected_files.loc[i]['filename'] 
        for mask_type in mask_types:
            for window_size in window_sizes: 
                for mask_percentage in mask_percentages:
                    mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
                                                mask_percentage=mask_percentage, 
                                                window_size=window_size,
                                                num_samples=NUM_SAMPLES, 
                                                function='euclidean',
                                                mask_type=mask_type)  
                    
                    id = int(selected_files.loc[i]['event_label'])
                    filename = selected_files.loc[i]['filename']
                    generate_explanation(
                        filename=filename,
                        model_name='ast',
                        id_to_explain=id,
                        config=mask_config,
                        path='/home/ec2-user/results1/explanations_audioset'
                    )
        for config in mask_configs:
            id = int(selected_files.loc[i]['event_label'])
            name=list(config.keys())[0]
            second_path = f"/home/ec2-user/results1/explanations_audioset/{filename}/ast/ft2_{id}_{name}.json"
            if os.path.exists(second_path):
                continue
            random_data = RandomDataGenerator(
                path='/home/ec2-user/results1/explanations_audioset', 
                model_name='ast',
                filename=filename,
                config=config,
                num_samples=NUM_SAMPLES, 
            )
            random_data.process_random_file()
            generate_explanation_from_file(filename, 
                        model_name='ast', 
                        id_to_explain=id,
                        name=name,
                        path='/home/ec2-user/results1/explanations_audioset')




