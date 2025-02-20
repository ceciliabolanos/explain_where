import pandas as pd
from tqdm import tqdm
from run_explanation import generate_explanation, generate_explanation_from_file
from data_generator.base_generator import MaskingConfig
from data_generator.random_generator import RandomDataGenerator
from utils import calculate_std

LABELS_PATH = '/home/ec2-user/Datasets/Audioset/labels/audioset_eval_train.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 2000
df = pd.read_csv(LABELS_PATH)

# Hyperparameters
mask_percentages = [0.3, 0.4, 0.2]
window_sizes = [5, 1, 3]
mask_types = ['noise', 'zeros']

selected_files = pd.read_csv('/home/ec2-user/explain_where/datasets/audioset/audioset.csv')

mask_configs = [ 
    {"zeros": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros'], "combinations": 9}},
    {"noise": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "combinations": 9}}
]


std = calculate_std('audioset')
print(std)
for mask_type in mask_types:
    for window_size in window_sizes: 
        for mask_percentage in mask_percentages:
            mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
                                        mask_percentage=mask_percentage, 
                                        window_size=window_size,
                                        num_samples=NUM_SAMPLES, 
                                        function='euclidean',
                                        mask_type=mask_type)  
            for i in tqdm(range(len(selected_files))):
                filename = selected_files.loc[i]['filename'] 
                id = int(selected_files.loc[i]['event_label'])
                try:
                    generate_explanation(
                        filename=filename,
                        model_name='ast',
                        id_to_explain=id,
                        config=mask_config,
                        std_dataset=std,
                        path='/home/ec2-user/results/explanations_audioset'
                    )
                except Exception as e:
                    print(f"Error generating explanation for {filename}: {e}")
    
# mask_configs = [ 
#     {"zeros": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros'], "combinations": 9}},
#     {"noise": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "combinations": 9}},
# ]

# for i in tqdm(range(len(selected_files))):
#     for config in mask_configs:
#         filename = selected_files.loc[i]['filename'] 
#         id = int(selected_files.loc[i]['event_label'])
#             random_data = RandomDataGenerator(
#                 path='/home/ec2-user/results/explanations_audioset', 
#                 model_name='ast',
#                 filename=filename,
#                 config=config,
#                 num_samples=3000, 
#             )
#             name=list(config.keys())[0]
#             random_data.process_random_file()
#             generate_explanation_from_file(filename, 
#                       model_name='ast', 
#                       id_to_explain=id,
#                       name=name,
#                       path='/home/ec2-user/results/explanations_audioset')
#             print('Done')




