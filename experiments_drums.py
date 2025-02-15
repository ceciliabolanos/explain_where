import pandas as pd
from tqdm import tqdm
import torch
import sys

from run_explanation import generate_explanation, generate_explanation_from_file
from data.base_generator import MaskingConfig
from data.random_generator import RandomDataGenerator

LABELS_PATH = '/home/ec2-user/explain_where/models/drums/drums_dataset.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 3000
df = pd.read_csv(LABELS_PATH)


# mask_percentages = [0.4, 0.2, 0.3]
# window_sizes = [3, 5, 1]
# mask_types = ['stat', 'zeros', 'noise']

# # Redirect stdout to a log file
# log_file_path = '/home/ec2-user/results1/process_log_drums.txt'
# sys.stdout = open(log_file_path, 'w')

# print("Starting explanation generation for Drums model.")

# for mask_type in mask_types:
#     for window_size in window_sizes: 
#         for mask_percentage in mask_percentages:
#             mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
#                                         mask_percentage=mask_percentage, 
#                                         window_size=window_size,
#                                         num_samples=NUM_SAMPLES, 
#                                         function='euclidean',
#                                         mask_type=mask_type)
            
#             # Print the configuration being used
#             print(f"Using mask_type: {mask_type}, window_size: {window_size}, mask_percentage: {mask_percentage}")
            
#             j = 0
#             for i in tqdm(range(len(df))):
#                 filename = df.loc[i, 'filename']
#                 id = df.loc[i, 'num_kicks']
#                 if j < 50:
#                     try:
#                         # Print the explanation generation process
#                         print(f"Generating explanation for {filename} (Sample {i+1}/{len(df)})")
#                         generate_explanation(
#                             filename=filename,
#                             model_name='drums',
#                             id_to_explain=id,
#                             config=mask_config,
#                             path='/home/ec2-user/results1/explanations_drums'
#                         )
#                         print(f"Successfully generated explanation for {filename}")
#                     except Exception as e:
#                         # Print any error during explanation generation
#                         print(f"Error generating explanation for {filename}: {e}")
#                 print(f"Finished processing {filename} ({j}/{len(df)}) for mask_type: {mask_type}, window_size: {window_size}, mask_percentage: {mask_percentage}")

#                 torch.cuda.empty_cache()
#                 del filename
#                 j += 1
                
# # Print final message when process is complete
# print("Explanation generation process for Drums model completed.")

# # Close the log file when done
# sys.stdout.close()


mask_configs = [
    {"0.2-1": {"mask_percentage": [0.2], "possible_windows": [1], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    {"0.2-3": {"mask_percentage": [0.2], "possible_windows": [3], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    {"0.2-5": {"mask_percentage": [0.2], "possible_windows": [5], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    
    {"0.3-1": {"mask_percentage": [0.3], "possible_windows": [1], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    {"0.3-3": {"mask_percentage": [0.3], "possible_windows": [3], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    {"0.3-5": {"mask_percentage": [0.3], "possible_windows": [5], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    
    {"0.4-1": {"mask_percentage": [0.4], "possible_windows": [1], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    {"0.4-3": {"mask_percentage": [0.4], "possible_windows": [3], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    {"0.4-5": {"mask_percentage": [0.4], "possible_windows": [5], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 3}},
    
    {"zeros": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros'], "combinations": 9}},
    {"noise": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['noise'], "combinations": 9}},
    {"stat": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['stat'], "combinations": 9}},
    
    {"all": {"mask_percentage": [0.2, 0.3, 0.4], "possible_windows": [1,3,5], "possible_mask_types": ['zeros', 'stat', 'noise'], "combinations": 27}},
]

for i in tqdm(range(len(df))):
    for config in mask_configs:
        filename = df.loc[i, 'filename']
        id = df.loc[i, 'num_kicks']
        if j < 50:
            random_data = RandomDataGenerator(
                path='/home/ec2-user/results1/explanations_drums', 
                model_name='drums',
                filename=filename,
                config=config,
                num_samples=NUM_SAMPLES, 
            )
            name=list(config.keys())[0]
            random_data.process_random_file()
            generate_explanation_from_file(filename, 
                      model_name='drums', 
                      id_to_explain=id,
                      name=name,
                      path='/home/ec2-user/results1/explanations_drums')
            print('Done')
    j=j+1
