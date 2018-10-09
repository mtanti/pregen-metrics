import numpy as np
import sys
import json
import random

import lib
import model_base
import model_normal
import data
import helper_datasources
import config

sys.path.append(config.mscoco_dir)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.wmd.wmd import WMD

_meteor_scorer = Meteor()
_cider_scorer  = Cider()
_spice_scorer  = Spice()
_wmd_scorer    = WMD()

########################################################################################
def geomean(xs):
    if np.all(xs): #If array does not contain a zero
        return 2**np.mean(np.log2(xs))
    else:
        return 0.0
    
########################################################################################
def get_meteor(test_tokenized_grouped_sents, generated):
    return _meteor_scorer.compute_score(
            {i: [ ' '.join(t) for t in ts ] for (i, ts) in enumerate(test_tokenized_grouped_sents)},
            {i: [ ' '.join(g) for g in gs ] for (i, gs) in enumerate(generated)}
        )[0]

########################################################################################
def get_cider(test_tokenized_grouped_sents, generated):
    return _cider_scorer.compute_score(
            {i: [ ' '.join(t) for t in ts ] for (i, ts) in enumerate(test_tokenized_grouped_sents)},
            {i: [ ' '.join(g) for g in gs ] for (i, gs) in enumerate(generated)}
        )[0]
        
########################################################################################
def get_individual_cider(test_tokenized_grouped_sents, generated):
    return _cider_scorer.compute_score(
            {i: [ ' '.join(t) for t in ts ] for (i, ts) in enumerate(test_tokenized_grouped_sents)},
            {i: [ ' '.join(g) for g in gs ] for (i, gs) in enumerate(generated)}
        )[1]
        
########################################################################################
def get_spice(test_tokenized_grouped_sents, generated):
    return _spice_scorer.compute_score(
            {i: [ ' '.join(t) for t in ts ] for (i, ts) in enumerate(test_tokenized_grouped_sents)},
            {i: [ ' '.join(g) for g in gs ] for (i, gs) in enumerate(generated)}
        )[0]
            
########################################################################################
def get_wmd(test_tokenized_grouped_sents, generated):
    return _wmd_scorer.compute_score(
            {i: [ ' '.join(t) for t in ts ] for (i, ts) in enumerate(test_tokenized_grouped_sents)},
            {i: [ ' '.join(g) for g in gs ] for (i, gs) in enumerate(generated)}
        )[0]

########################################################################################
def filter_none(token_probs):
    return [prob for (prob, rank) in token_probs]
    
########################################################################################
def filter_matching_tokens(token_probs):
    return [prob for (prob, rank) in token_probs if rank != 0]
    
########################################################################################
def take_matching_prefix(token_probs):
    probs = []
    for (prob, rank) in token_probs:
        if rank != 0:
            break
        else:
            probs.append(prob)
    return probs

########################################################################################
def cap_scorer_prob(token_probs):
    if len(token_probs) > 0:
        if np.all(token_probs): #If array does not contain a zero
            return np.prod(token_probs)
        else:
            return 0.0
    else:
        return 0.0

########################################################################################
def cap_scorer_pplx(token_probs):
    if len(token_probs) > 0:
        if np.all(token_probs): #If array does not contain a zero
            return 2**-np.mean(np.log2(token_probs))
        else:
            return np.inf
    else:
        return np.inf

########################################################################################
########################################################################################
########################################################################################
postgens = [
        ('METEOR', get_meteor),
        ('CIDEr', get_cider),
        ('SPICE', get_spice),
        ('WMD', get_wmd),
    ]
    
dataset_level_aggrs = [
        ('sum',     np.sum),
        ('mean',    np.mean),
        ('median',  np.median),
        ('geomean', geomean),
        ('max',     np.max),
        ('min',     np.min),
    ]
image_level_aggrs = [
        ('sum',     lambda dataset:[np.sum(img) for img in dataset]),
        ('mean',    lambda dataset:[np.mean(img) for img in dataset]),
        ('median',  lambda dataset:[np.median(img) for img in dataset]),
        ('geomean', lambda dataset:[geomean(img) for img in dataset]),
        ('max',     lambda dataset:[np.max(img) for img in dataset]),
        ('min',     lambda dataset:[np.min(img) for img in dataset]),
        ('join',    lambda dataset:[cap for img in dataset for cap in img]),
    ]
caption_scorers = [
        ('prob',      lambda dataset:[[cap_scorer_prob(cap) for (cap, orig_len) in img] for img in dataset]),
        ('pplx',      lambda dataset:[[cap_scorer_pplx(cap) for (cap, orig_len) in img] for img in dataset]),
        ('count',     lambda dataset:[[len(cap) for (cap, orig_len) in img] for img in dataset]),
        ('normcount', lambda dataset:[[len(cap)/float(orig_len) for (cap, orig_len) in img] for img in dataset]),
    ]
rank_filters = [
        ('none',    lambda dataset:[[(filter_none(cap), len(cap)) for cap in img] for img in dataset]),
        ('filter0', lambda dataset:[[(filter_matching_tokens(cap), len(cap)) for cap in img] for img in dataset]),
        ('prefix0', lambda dataset:[[(take_matching_prefix(cap), len(cap)) for cap in img] for img in dataset]),
    ]

with open('results/data.txt', 'w', encoding='utf-8') as f:
    print(
            'dataset', 'architecture', 'run', 'num_strata', 'stratum',
            
            *[postgen for (postgen, _) in postgens],
            
            *[
                '_'.join([dataset_level_aggr, image_level_aggr, caption_scorer, rank_filter])
                for (dataset_level_aggr, _) in dataset_level_aggrs
                for (image_level_aggr, _) in image_level_aggrs
                for (caption_scorer, _) in caption_scorers
                for (rank_filter, _) in rank_filters
            ],
            
            sep='\t', file=f
        )
    
rand = random.Random()
for dataset_name in [ 'flickr8k', 'flickr30k', 'mscoco' ]:
    datasources = helper_datasources.DataSources(dataset_name)
    for architecture in [ 'init', 'pre', 'par', 'merge' ]:
        for run in range(1, config.num_runs+1):
            print()
            print('='*100)
            print('{}_{}_{}'.format(dataset_name, architecture, run))
            
            with open('model_data/{}_{}_{}/generated_captions.txt'.format(dataset_name, architecture, run), 'r', encoding='utf-8') as f:
                generated_caps = [ [line.split(' ')] for line in f.read().strip().split('\n') ]
            with open('test_probs/{}_{}_{}.json'.format(dataset_name, architecture, run), 'r', encoding='utf-8') as f:
                test_probs = json.load(f)
            
            sorted_indexes = np.argsort(get_individual_cider(datasources.test.caption_groups, generated_caps))
            num_imgs = len(sorted_indexes)
            for num_strata in range(1, 5+1):
                for stratum in range(num_strata):
                    selected_test_caps = [ datasources.test.caption_groups[i] for i in sorted_indexes[num_imgs//num_strata*stratum:num_imgs//num_strata*(stratum+1)] ]
                    selected_generated_caps = [ generated_caps[i] for i in sorted_indexes[num_imgs//num_strata*stratum:num_imgs//num_strata*(stratum+1)] ]
                    selected_test_probs = [ test_probs[i] for i in sorted_indexes[num_imgs//num_strata*stratum:num_imgs//num_strata*(stratum+1)] ]
                
                    with open('results/data.txt', 'a', encoding='utf-8') as f:
                        print(
                                dataset_name, architecture, run, num_strata, stratum,
                                
                                *[postgen(selected_test_caps, selected_generated_caps) for (_, postgen) in postgens],
                                
                                *[
                                    dataset_level_aggr(image_level_aggr(caption_scorer(rank_filter(selected_test_probs))))
                                    for (_, dataset_level_aggr) in dataset_level_aggrs
                                    for (_, image_level_aggr) in image_level_aggrs
                                    for (_, caption_scorer) in caption_scorers
                                    for (_, rank_filter) in rank_filters
                                ],
                                
                                sep='\t', file=f
                            )