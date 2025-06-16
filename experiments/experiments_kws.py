import pandas as pd
from tqdm import tqdm
import torch

from run_explanation import generate_explanation, generate_explanation_from_file
from data_generator.base_generator import MaskingConfig
from data_generator.random_generator import RandomDataGenerator
from utils import calculate_std

LABELS_PATH = '/home/ec2-user/explain_where/datasets/kws/kws_dataset.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 3000
df = pd.read_csv(LABELS_PATH)

mask_percentages = [0.2, 0.4, 0.3]
window_sizes = [1, 5, 3]
mask_types = ['zeros', 'noise']

# std = calculate_std('kws')
std = 0.06
print(std)

for mask_type in mask_types:
    for window_size in window_sizes: 
        for mask_percentage in mask_percentages:
            mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
                                        mask_percentage=mask_percentage, 
                                        window_size=window_size,
                                        num_samples=NUM_SAMPLES, 
                                        function='euclidean',
                                        mask_type=mask_type, 
                                        std_noise=std)
            
            for i in tqdm(range(len(df))):
                filename = df.loc[i, 'filename']
                if i < 50:
                    try:
                        generate_explanation(
                            filename=filename,
                            model_name='kws',
                            id_to_explain=0,
                            config=mask_config,
                            path='/home/ec2-user/results/explanations_kws'
                        )
                    except Exception as e:
                        print(f"Error generating explanation for {filename}: {e}")
            
                torch.cuda.empty_cache()
                del filename            

samples = [100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12000, 14000, 18000]
mask_configs = [
    {"zeros": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5],"possible_std_noise": [0.06], "possible_mask_types": ['zeros'], "combinations": 9}},
    {"noise": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_std_noise": [0.06], "possible_mask_types": ['noise'], "combinations": 9}},
]

j = 0
for i in tqdm(range(len(df))):
    for config in mask_configs:
        for sample in samples:
            filename = df.loc[i, 'filename']
            if i < 50:
                random_data = RandomDataGenerator(
                    path='/home/ec2-user/results/explanations_kws', 
                    model_name='kws',
                    filename=filename,
                    config=config,
                    num_samples=sample, 
                )
                name=list(config.keys())[0]
                random_data.process_random_file()
                generate_explanation_from_file(filename, 
                        model_name='kws', 
                        id_to_explain=0,
                        name=name,
                        num_samples=sample,
                        path='/home/ec2-user/results/explanations_kws')
                print('Done')
        # j=j+1

