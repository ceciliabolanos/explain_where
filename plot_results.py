import numpy as np
import pandas as pd
import itertools
from confidence_intervals.utils import barplot_with_ci
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import sys
import re, ast
from IPython import embed
from utils import barplot_with_ci, read_results_file, mean_with_confint, map_metric_name

metric = 'top50'
outfile = 'top50.png'

metric_fieldname, metric_name_in_tsv, intersection = map_metric_name(metric)

# Dictionaries to define the names to be used in the plots
dataset_dict = {'drums': 'Drums', 'kws': 'KWS', 'audioset': 'Audioset', 'audioset_speech': 'Speech', 'audioset_dog': 'Dog', 'audioset_music': 'Music'}
metric_dict = {'auc': 'Average AUC', 'auc_relaxed': 'Average AUC', 'leo_metric': 'LM'}
names_dict = OrderedDict([('noise','noise'), ('zeros','zero')]) #, ('stat', 'mean'), ('all', 'mix')])
methods_dict = OrderedDict([('RF', 'RF'), ('LR','LR'), ('SHAP','SHAP')])

methods = list(methods_dict.keys())
names   = list(names_dict.keys())

# datasets = ['drums', 'kws', 'audioset_music', 'audioset_speech', 'audioset_dog']
datasets = ['kws']

fig, axs = plt.subplots(1, len(datasets), figsize=(7,3), sharey=True)
axs = np.atleast_1d(axs)  # Convert to array if it's not

for dataseti, dataset in enumerate(datasets):

    base_path = f'/home/ec2-user/evaluations/{dataset}'

    files_auc_complete = [f'{base_path}/{metric_name_in_tsv}_{method}_{{}}{intersection}.tsv' for method in methods]

    data2plot = OrderedDict()
    for name in names:
        data2plot[names_dict[name]] = OrderedDict()
        for method, file_template in zip(methods, files_auc_complete):
            file_path = file_template.format(name)
            #print(file_path)
            df_combination = read_results_file(file_path, metric, method, name, dataset)
            values = df_combination[metric_fieldname]
            perf = mean_with_confint(values)
            data2plot[names_dict[name]][methods_dict[method]] = perf
            print(f'{dataset} {name} {method} {perf[0]:4f}')

    metric_name = metric_dict[metric] if metric in metric_dict else metric
    dataset_name = dataset_dict[dataset]
    barplot_with_ci(axs[dataseti], data2plot, dataset_name, metric_name, legend=dataseti==0)


plt.tight_layout()
plt.savefig(outfile)