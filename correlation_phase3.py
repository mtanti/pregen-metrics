import numpy as np
import sys
import json
import pandas as pd
from scipy.stats.stats import pearsonr

import lib
import model_base
import model_normal
import data
import helper_datasources
import config

postgens = [
        'METEOR',
        'CIDEr',
        'SPICE',
        'WMD',
    ]
    
dataset_level_aggrs = [
        'sum',
        'mean',
        'median',
        'geomean',
        'max',
        'min',
    ]
image_level_aggrs = [
        'sum',
        'mean',
        'median',
        'geomean',
        'max',
        'min',
        'join',
    ]
caption_scorers = [
        'prob',
        'pplx',
        'count',
        'normcount',
    ]
rank_filters = [
        'none',
        'filter0',
        'prefix0',
    ]
    
########################################################################################
dframe = pd.read_csv('results/data.txt', sep='\t')
for postgen in postgens:
    print()
    print('='*100)
    print(postgen)
    with open('results/correlations_{}.txt'.format(postgen), 'w', encoding='utf-8') as f:
        print('pregen', *[
                            '{}_{}'.format(dataset_name, measure)
                            for dataset_name in ['mscoco', 'flickr30k', 'flickr8k', 'all']
                            for measure in ['R^2', 'R', 'p']
                        ], sep='\t', file=f)
        for pregen in [
                    '_'.join([dataset_level_aggr, image_level_aggr, caption_scorer, rank_filter])
                    for dataset_level_aggr in dataset_level_aggrs
                    for image_level_aggr in image_level_aggrs
                    for caption_scorer in caption_scorers
                    for rank_filter in rank_filters
                ]:
            print(pregen)
            
            try:
                results = list()
                for dataset_name in ['mscoco', 'flickr30k', 'flickr8k', 'all']:
                    if dataset_name != 'all':
                        filtered_indexes = dframe['dataset'] == dataset_name
                    else:
                        filtered_indexes = dframe['dataset'] == dframe['dataset']
                    if np.any(np.isinf(dframe[pregen][filtered_indexes])):
                        #print('', 'infs')
                        raise StopIteration()
                    if np.max(dframe[pregen][filtered_indexes]) - np.min(dframe[pregen][filtered_indexes]) == 0.0:
                        #print('', 'equals')
                        raise StopIteration()
                    (R, p) = pearsonr(dframe[postgen][filtered_indexes], dframe[pregen][filtered_indexes])
                    results.extend([R**2, R, p])
                print(pregen, *results, sep='\t', file=f)
            except StopIteration:
                pass