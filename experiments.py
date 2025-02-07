import pandas as pd
from tqdm import tqdm
from run_explanation import generate_explanation
from data.base_generator import MaskingConfig
import os 

LABELS_PATH = '/home/ec2-user/Datasets/Audioset/labels/audioset_eval_train.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 3000

# Masking parameters to try

# functions = ['euclidean', 'cosine', 'dtw']
# mask_percentages = [0.1, 0.2, 0.3, 0.4]
# window_sizes = [1, 2, 3, 4, 5]
# mask_types = ['zeros', 'stat', 'noise']

functions = ['euclidean', 'cosine', 'dtw']
mask_percentages = [0.1, 0.2, 0.3, 0.4]
window_sizes = [1, 2, 3, 4, 5]
mask_types = ['zeros']
df = pd.read_csv(LABELS_PATH)

for function in functions:
    for mask_type in mask_types:
        for window_size in window_sizes: 
            for mask_percentage in mask_percentages:
                mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
                                            mask_percentage=mask_percentage, 
                                            window_size=window_size,
                                            num_samples=NUM_SAMPLES, 
                                            function=function,
                                            mask_type=mask_type)
                j =0
                for i in tqdm(range(len(df))):
                    filename = df.loc[i, 'base_segment_id']
                    id = df.loc[i, 'father_id_ast']
                    if j < 138:
                        generate_explanation(
                            filename=filename,
                            model_name='ast',
                            id_to_explain=id,
                            config=mask_config,
                        )
                    j=j+1
