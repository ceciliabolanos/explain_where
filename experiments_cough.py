import pandas as pd
from tqdm import tqdm
import torch
from utils import calculate_std
from run_explanation import generate_explanation, generate_explanation_from_file

from data_generator.base_generator import MaskingConfig
from data_generator.random_generator import RandomDataGenerator

LABELS_PATH = '/home/ec2-user/explain_where/datasets/cough/cough_happy.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 400

mask_percentages = [0.2, 0.3, 0.4]
window_sizes = [5, 1, 3]
mask_types = ['zeros', 'noise']

df = pd.read_csv(LABELS_PATH)

# std = calculate_std('cough')
std = 0.135
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
            j = 0   
            for i in tqdm(range(len(df))):
                filename = df.loc[i, 'filename']
                if j < 50:
                    try:
                        generate_explanation(
                            filename=filename,
                            model_name='cough',
                            id_to_explain=1,
                            config=mask_config,
                            std_dataset=std,
                            path='/home/ec2-user/results/explanations_cough'
                        )
                    except Exception as e:
                        print(f"Error generating explanation for {filename}: {e}")
            
                torch.cuda.empty_cache()
                del filename
                j = j + 1


# mask_configs = [ 
#     {"zeros": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros'], "combinations": 9}},
#     {"noise": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "combinations": 9}}
# ]

# j=0
# for i in tqdm(range(len(df))):
#     filename = df.loc[i, 'filename']
#     for config in mask_configs:
#         name=list(config.keys())[0]
#         if j < 50:
#             random_data = RandomDataGenerator(
#                 path='/home/ec2-user/results/explanations_cough', 
#                 model_name='cough',
#                 filename=filename,
#                 config=config,
#                 num_samples=3000, 
#             )
#             random_data.process_random_file()
#             generate_explanation_from_file(filename, 
#                         model_name='cough', 
#                         id_to_explain=1,
#                         name=name,
#                         path='/home/ec2-user/results/explanations_cough')
#     j=j+1
            