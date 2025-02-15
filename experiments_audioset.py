import pandas as pd
from tqdm import tqdm
from run_explanation import generate_explanation
from data.base_generator import MaskingConfig
import json
from multiprocessing import Pool, cpu_count
from itertools import product
import sys

LABELS_PATH = '/home/ec2-user/Datasets/Audioset/labels/audioset_eval_train.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 3000

# Hyperparameters
mask_percentages = [0.3, 0.4, 0.1, 0.2]
window_sizes = [5, 1, 3]
mask_types = ['noise','stat','zeros']
df_chunk = pd.read_csv(LABELS_PATH)

# Redirect stdout to a log file
# log_file_path = '/home/ec2-user/results1/process_log_audioset.txt'
# sys.stdout = open(log_file_path, 'w')

with open('/home/ec2-user/explain_where/preprocess/selected_files.json', 'r') as f:
    selected_files = json.load(f)

j = 0
for mask_type in mask_types:
    for window_size in window_sizes: 
        for mask_percentage in mask_percentages:
            mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
                                        mask_percentage=mask_percentage, 
                                        window_size=window_size,
                                        num_samples=NUM_SAMPLES, 
                                        function='euclidean',
                                        mask_type=mask_type)
            
            j = 0
            for i, row in tqdm(enumerate(df_chunk.itertuples())):
                id = row.father_id_ast
                filename = row.base_segment_id
                if (str(id) in selected_files.keys()) & (j <70):
                    if filename in selected_files[f'{id}']:
                        j += 1
                        generate_explanation(
                            filename=filename,
                            model_name='ast',
                            id_to_explain=id,
                            config=mask_config,
                            path='/home/ec2-user/results1/explanations_audioset'
                        )

# mask_configs = [
#     {"0.2-1": {"mask_percentage": [0.2], "possible_windows": [1], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
#     {"0.2-3": {"mask_percentage": [0.2], "possible_windows": [3], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
#     {"0.2-5": {"mask_percentage": [0.2], "possible_windows": [5], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    
#     {"0.3-1": {"mask_percentage": [0.3], "possible_windows": [1], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
#     {"0.3-3": {"mask_percentage": [0.3], "possible_windows": [3], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
#     {"0.3-5": {"mask_percentage": [0.3], "possible_windows": [5], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    
#     {"0.4-1": {"mask_percentage": [0.4], "possible_windows": [1], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
#     {"0.4-3": {"mask_percentage": [0.4], "possible_windows": [3], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
#     {"0.4-5": {"mask_percentage": [0.4], "possible_windows": [5], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    
#     {"zeros": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros'], "combinations": 9}},
#     {"noise": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "combinations": 9}},
#     {"stat": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['stat'], "combinations": 9}},
    
#     {"all": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 27}},
# ]

#for i, row in tqdm(enumerate(df_chunk.itertuples())):
#   for config in mask_configs:
#         if i < 240:
#             random_data = RandomDataGenerator(
#                 path='/home/ec2-user/results1/explanations_audioset', 
#                 model_name='ast',
#                 filename=row.base_segment_id,
#                 config=config,
#                 num_samples=NUM_SAMPLES, 
#             )
#             name=list(config.keys())[0]
#             random_data.process_random_file()
#             generate_explanation_from_file(row.base_segment_id, 
#                       model_name='ast', 
#                       id_to_explain=row.father_id_ast,
#                       name=name,
#                       path='/home/ec2-user/results1/explanations_audioset')
#             print('Done')