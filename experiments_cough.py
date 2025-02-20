import pandas as pd
from tqdm import tqdm
import torch
import sys

from run_explanation import generate_explanation, generate_explanation_from_file

from data.base_generator import MaskingConfig
from data.random_generator import RandomDataGenerator

LABELS_PATH = '/home/ec2-user/explain_where/models/cough/cough_happy.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 400

mask_percentages = [0.2, 0.3, 0.4]
window_sizes = [5, 1, 3]
mask_types = ['zeros']

df = pd.read_csv(LABELS_PATH)

print("Starting explanation generation for KWS model.")
mask_configs = [ 
    {"zeros": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros'], "combinations": 9}},
    # {"noise": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "combinations": 9}}
]
for mask_type in mask_types:
    for window_size in window_sizes: 
        for mask_percentage in mask_percentages:
            mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
                                        mask_percentage=mask_percentage, 
                                        window_size=window_size,
                                        num_samples=NUM_SAMPLES, 
                                        function='euclidean',
                                        mask_type=mask_type)
            
            # Print the configuration being used
            print(f"Using mask_type: {mask_type}, window_size: {window_size}, mask_percentage: {mask_percentage}")
            j = 0   
            for i in tqdm(range(len(df))):
                filename = df.loc[i, 'filename']
                if j < 50:
                    try:
                        # Print the explanation generation process
                        print(f"Generating explanation for {filename} (Sample {i+1}/{len(df)})")
                        generate_explanation(
                            filename=filename,
                            model_name='cough',
                            id_to_explain=1,
                            config=mask_config,
                            path='/home/ec2-user/results1/explanations_cough'
                        )
                        print(f"Successfully generated explanation for {filename}")
                    except Exception as e:
                        # Print any error during explanation generation
                        print(f"Error generating explanation for {filename}: {e}")
            
                torch.cuda.empty_cache()
                del filename
                j = j + 1
            # Print progress in the loop
            
            print(f"Finished processing for mask_type: {mask_type}, window_size: {window_size}, mask_percentage: {mask_percentage}, j={j}")

print("Explanation generation process completed.")
for i in tqdm(range(len(df))):
    filename = df.loc[i, 'filename']
    for config in mask_configs:
        name=list(config.keys())[0]
        random_data = RandomDataGenerator(
            path='/home/ec2-user/results1/explanations_cough', 
            model_name='cough',
            filename=filename,
            config=config,
            num_samples=3000, 
        )
        random_data.process_random_file()
        generate_explanation_from_file(filename, 
                    model_name='cough', 
                    id_to_explain=1,
                    name=name,
                    path='/home/ec2-user/results1/explanations_cough')
        
# Close the log file when done


# j =0
# names = ['zeros_random', 'stat_random', 'noise_random', 'random']
# possible_types = [['zeros'], ['stat'], ['noise'], ['zeros', 'stat', 'noise']]

# for i in tqdm(range(len(df))):
#     for type, name in zip(possible_types, names):
#         filename = df.loc[i, 'filename']
#         if j < 50:
#             random_data = RandomDataGenerator(
#                 path='/home/ec2-user/results1/explanations_cough', 
#                 model_name='cough',
#                 filename=filename,
#                 windows=window_sizes,
#                 functions=['euclidean'],
#                 mask_types=type,
#                 mask_percentages=mask_percentages,
#                 num_samples=NUM_SAMPLES, 
#                 name=name
#             )
#             random_data.process_random_file()
#             generate_explanation_from_file(filename, 
#                       model_name='cough', 
#                       id_to_explain=1,
#                       name=name,
#                       path='/home/ec2-user/results1/explanations_cough')
#     j=j+1
