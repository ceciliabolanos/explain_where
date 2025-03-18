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
mask_percentages = [0.2, 0.3, 0.4]
window_sizes = [3, 5, 1]
mask_types = ['noise']
alphas = [0.1, 0.5, 1]

selected_files = pd.read_csv('/home/ec2-user/explain_where/datasets/audioset/audioset.csv')

# std = calculate_std('audioset')
# std=0.11 # ya lo habiamos calculado antes
# print(std)
for mask_type in mask_types:
    for window_size in window_sizes: 
        for mask_percentage in mask_percentages:
            for alpha in alphas:
                mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
                                            mask_percentage=mask_percentage, 
                                            window_size=window_size,
                                            num_samples=NUM_SAMPLES, 
                                            function='euclidean',
                                            mask_type=mask_type,
                                            std_noise=alpha)  
                for i in tqdm(range(len(selected_files))):
                    if i < 140:
                        filename = selected_files.loc[i]['filename'] 
                        id = int(selected_files.loc[i]['event_label'])
                        try:
                            generate_explanation(
                                filename=filename,
                                model_name='ast',
                                id_to_explain=id,
                                config=mask_config,
                                path='/home/ec2-user/results/explanations_audioset'
                            )
                        except Exception as e:
                            print(f"Error generating explanation for {filename}: {e}")
            
# mask_configs = [ 
#     {"zeros": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros'], "possible_std_noise": [0],"combinations": 9}},
#     {"noise_0.1": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "possible_std_noise": [0.1], "combinations": 9}},
#     {"noise_0.5": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "possible_std_noise": [0.5], "combinations": 9}},
#     {"noise_1": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "possible_std_noise": [1], "combinations": 9}},
# ]

samples = [100, 200, 400, 600, 800, 1000, 1500, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 18000]

# for i in tqdm(range(len(selected_files))):
#     if i < 140:
#         for config in mask_configs:
#             for sample in samples:
#                 filename = selected_files.loc[i]['filename'] 
    #             id = int(selected_files.loc[i]['event_label'])
    #             random_data = RandomDataGenerator(
    #                 path='/home/ec2-user/results/explanations_audioset', 
    #                 model_name='ast',
    #                 filename=filename,
    #                 config=config,
    #                 num_samples=sample, 
    #             )
    #             name=list(config.keys())[0]
    #             random_data.process_random_file()
    #             generate_explanation_from_file(filename, 
    #                         model_name='ast', 
    #                         id_to_explain=id,
    #                         name=name,
    #                         path='/home/ec2-user/results/explanations_audioset')
    #             print('Done')




