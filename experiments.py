import pandas as pd
from tqdm import tqdm
from run_explanation import generate_explanation
from data.base_generator import MaskingConfig
import os 

LABELS_PATH = '/home/ec2-user/Datasets/Audioset/labels/audioset_eval_train.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 3000

# Masking parameters to try
mask_percentages = [0.1, 0.2, 0.3, 0.4]
window_sizes = [1, 2, 3, 4, 5]
functions = ['euclidean', 'cosine', 'dtw'] 
mask_types = ['zeros', 'noise', 'stat']

df = pd.read_csv(LABELS_PATH)

for mask_type in mask_types:
    for function in functions:     
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
                    if j < 138 and (not os.path.exists(f'/home/ec2-user/results1/explanations_audioset/{df.loc[i, "base_segment_id"]}/ast/scores_w{window_size}_m{mask_type}.json')):
                        generate_explanation(
                            filename=df.loc[i, 'base_segment_id'],
                            model_name='ast',
                            id_to_explain=df.loc[i, 'father_id_ast'],
                            config=mask_config,
                        )
                    j=j+1
