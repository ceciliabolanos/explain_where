import pandas as pd
from tqdm import tqdm
from run_explanation import generate_explanation
from data.base_generator import MaskingConfig
import os 
import torch

LABELS_PATH = '/home/cbolanos/explain_where/models/cough/cough_happy.csv'
SEGMENT_LENGTH = 100
NUM_SAMPLES = 3000

mask_percentages = [0.2, 0.3, 0.4]
window_sizes = [5, 1, 3]
mask_types = ['stat', 'noise', 'zeros']

df = pd.read_csv(LABELS_PATH)

for mask_type in mask_types:
    for window_size in window_sizes: 
        for mask_percentage in mask_percentages:
            mask_config = MaskingConfig(segment_length=SEGMENT_LENGTH, 
                                        mask_percentage=mask_percentage, 
                                        window_size=window_size,
                                        num_samples=NUM_SAMPLES, 
                                        function='euclidean',
                                        mask_type=mask_type)
            j =0
            for i in tqdm(range(len(df))):
                filename = df.loc[i, 'filename']
                if j < 50:
                    generate_explanation(
                        filename=filename,
                        model_name='cough',
                        id_to_explain=1,
                        config=mask_config,
                        path='/home/cbolanos/results1/explanations_cough'
                    )
                torch.cuda.empty_cache()
                del filename
                j=j+1
