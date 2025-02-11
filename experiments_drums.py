import pandas as pd
from tqdm import tqdm
import torch

from run_explanation import generate_explanation, generate_explanation_from_file

from data.base_generator import MaskingConfig
from data.random_generator import RandomDataGenerator


LABELS_PATH = '/home/cbolanos/explain_where/models/drums/drums_dataset.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 3000

mask_percentages = [0.2, 0.3, 0.4]
window_sizes = [1, 3, 5]
mask_types = ['noise', 'zeros', 'stat']

df = pd.read_csv(LABELS_PATH)

# for mask_type in mask_types:
#     for window_size in window_sizes: 
#         for mask_percentage in mask_percentages:
#             mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
#                                         mask_percentage=mask_percentage, 
#                                         window_size=window_size,
#                                         num_samples=NUM_SAMPLES, 
#                                         function='euclidean',
#                                         mask_type=mask_type)
#             j =0
#             for i in tqdm(range(len(df))):
#                 filename = df.loc[i, 'filename']
#                 id = df.loc[i, 'num_kicks']
#                 if j < 50:
#                     generate_explanation(
#                         filename=filename,
#                         model_name='drums',
#                         id_to_explain=id,
#                         config=mask_config,
#                         path='/home/cbolanos/results1/explanations_drums'
#                     )
#                 torch.cuda.empty_cache()
#                 del filename
#                 j=j+1



j =0
names = ['zeros_random', 'stat_random', 'noise_random', 'random']
possible_types = [['zeros'], ['stat'], ['noise'], ['zeros', 'stat', 'noise']]

for i in tqdm(range(len(df))):
    for type, name in zip(possible_types, names):
        filename = df.loc[i, 'filename']
        id = df.loc[i, 'num_kicks']
        if j < 50:
            random_data = RandomDataGenerator(
                path='/home/cbolanos/results1/explanations_drums', 
                model_name='drums',
                filename=filename,
                windows=window_sizes,
                functions=['euclidean'],
                mask_types=type,
                mask_percentages=mask_percentages,
                num_samples=NUM_SAMPLES, 
                name=name
            )
            random_data.process_random_file()
            generate_explanation_from_file(filename, 
                      model_name='drums', 
                      id_to_explain=id,
                      name=name,
                      path='/home/cbolanos/results1/explanations_drums')
        j=j+1
