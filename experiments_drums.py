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

mask_percentages = [0.4, 0.2, 0.3]
window_sizes = [3, 5, 1]
mask_types = ['stat', 'zeros', 'noise']

df = pd.read_csv(LABELS_PATH)

# Redirect stdout to a log file
log_file_path = '/home/ec2-user/results1/process_log_drums.txt'
sys.stdout = open(log_file_path, 'w')

print("Starting explanation generation for Drums model.")

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
                id = df.loc[i, 'num_kicks']
                if j < 50:
                    try:
                        # Print the explanation generation process
                        print(f"Generating explanation for {filename} (Sample {i+1}/{len(df)})")
                        generate_explanation(
                            filename=filename,
                            model_name='drums',
                            id_to_explain=id,
                            config=mask_config,
                            path='/home/ec2-user/results1/explanations_drums'
                        )
                        print(f"Successfully generated explanation for {filename}")
                    except Exception as e:
                        # Print any error during explanation generation
                        print(f"Error generating explanation for {filename}: {e}")
                print(f"Finished processing {filename} ({j}/{len(df)}) for mask_type: {mask_type}, window_size: {window_size}, mask_percentage: {mask_percentage}")

                torch.cuda.empty_cache()
                del filename
                j += 1
                
# Print final message when process is complete
print("Explanation generation process for Drums model completed.")

# Close the log file when done
sys.stdout.close()

# j =0
# names = ['zeros_random', 'stat_random', 'noise_random', 'random']
# possible_types = [['zeros'], ['stat'], ['noise'], ['zeros', 'stat', 'noise']]

# for i in tqdm(range(len(df))):
#     for type, name in zip(possible_types, names):
#         filename = df.loc[i, 'filename']
#         id = df.loc[i, 'num_kicks']
#         if j < 50:
#             random_data = RandomDataGenerator(
#                 path='/home/ec2-user/results1/explanations_drums', 
#                 model_name='drums',
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
#                       model_name='drums', 
#                       id_to_explain=id,
#                       name=name,
#                       path='/home/ec2-user/results1/explanations_drums')
#     j=j+1
