import numpy as np
import json
import collections
import os

import lib
import model_base
import model_normal
import data
import helper_datasources
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_name = 'mscoco'
architecture = 'init'
run = 1
beam_width = 1
lower_bound_len = 5
upper_bound_len = 50

def pregen(model, imgs, refs):
    num_batched_caps = model.val_minibatch_size
    
    caption_groups_probs = collections.defaultdict(list)
    
    batch_img_ids = []
    unpadded_batch_caps = []
    batch_caps_lens = []
    batch_images = []
    amount = sum(len(cap_group) for cap_group in refs)
    i = 0
    for (img_id, (cap_group, img)) in enumerate(zip(refs, imgs)):
        for cap in cap_group:
            i += 1
            batch_img_ids.append(img_id)
            unpadded_batch_caps.append([ model.dataset.token_to_index.get(token, data.UNKNOWN_INDEX) for token in cap ])
            batch_caps_lens.append(len(cap)+1) #include edge
            batch_images.append(img)
            if len(unpadded_batch_caps) == num_batched_caps or i == amount: #if batch is full or all captions have been processed
                max_len = max(batch_caps_lens)
                batch_caps = np.zeros([len(unpadded_batch_caps), max_len], np.int32)
                batch_targets = np.zeros([len(unpadded_batch_caps), max_len], np.int32)
                for (j, (indexes, cap_len)) in enumerate(zip(unpadded_batch_caps, batch_caps_lens)):
                    batch_caps[j,:cap_len] = [data.EDGE_INDEX]+indexes
                    batch_targets[j,:cap_len] = indexes+[data.EDGE_INDEX]
                
                batch_distributions = model.get_raw_probs(batch_images, batch_caps, batch_caps_lens, 1.0)
                    
                for (img_id, distribution, targets, cap_len) in zip(batch_img_ids, batch_distributions, batch_targets, batch_caps_lens):
                    target_probs = distribution[np.arange(distribution.shape[0]), targets][:cap_len]
                    max_probs = [ token_probs.max() for token_probs in distribution ][:cap_len]
                    caption_groups_probs[img_id].append((target_probs, max_probs))
                
                del batch_img_ids[:]
                del unpadded_batch_caps[:]
                del batch_caps_lens[:]
                del batch_images[:]
                    
    #mean max normcount prefix0
    img_scores = []
    for group in caption_groups_probs.values():
        cap_scores = []
        for (target_probs, max_probs) in group:
            filtered_probs = list()
            for (target_prob, max_prob) in zip(target_probs, max_probs):
                if target_prob == max_prob:
                    filtered_probs.append(target_prob)
                else:
                    break
            normcount = len(filtered_probs)/float(len(target_probs))
            cap_scores.append(normcount)
        img_scores.append(np.max(cap_scores))
    return np.mean(img_scores)
    
print('loading dataset')

datasources = helper_datasources.DataSources(dataset_name)
dataset = data.Dataset(
        min_token_freq        = config.min_token_freq,
        training_datasource   = datasources.train,
        validation_datasource = datasources.val,
        testing_datasource    = datasources.test,
    )
dataset.process()

print('loading model')
with model_normal.NormalModel(
        dataset                 = dataset,
        init_method             = config.hyperparams[architecture]['init_method'],
        min_init_weight         = config.hyperparams[architecture]['min_init_weight'],
        max_init_weight         = config.hyperparams[architecture]['max_init_weight'],
        embed_size              = config.hyperparams[architecture]['embed_size'],
        rnn_size                = config.hyperparams[architecture]['rnn_size'],
        post_image_size         = config.hyperparams[architecture]['post_image_size'],
        post_image_activation   = config.hyperparams[architecture]['post_image_activation'],
        rnn_type                = config.hyperparams[architecture]['rnn_type'],
        learnable_init_state    = config.hyperparams[architecture]['learnable_init_state'],
        multimodal_method       = architecture,
        optimizer               = config.hyperparams[architecture]['optimizer'],
        learning_rate           = config.hyperparams[architecture]['learning_rate'],
        normalize_image         = config.hyperparams[architecture]['normalize_image'],
        weights_reg_weight      = config.hyperparams[architecture]['weights_reg_weight'],
        image_dropout_prob      = config.hyperparams[architecture]['image_dropout_prob'],
        post_image_dropout_prob = config.hyperparams[architecture]['post_image_dropout_prob'],
        embedding_dropout_prob  = config.hyperparams[architecture]['embedding_dropout_prob'],
        rnn_dropout_prob        = config.hyperparams[architecture]['rnn_dropout_prob'],
        max_epochs              = config.hyperparams[architecture]['max_epochs'] if not config.debug else 2,
        val_minibatch_size      = config.val_minibatch_size,
        train_minibatch_size    = config.hyperparams[architecture]['train_minibatch_size'],
    ) as m:
    m.compile_model()
    m.load_params('model_data/{}_{}_{}'.format(dataset_name, architecture, run))
    
    with open('results/timing.txt', 'w', encoding='utf-8') as f:
        ########################################################################################
        print('measuring pregen timing')
        print('pregen', file=f)
        print('dataset', 'architecture', 'run', 'pregen_time', file=f)
        
        sub_timer = lib.Timer()
        
        pregen_score = pregen(m, datasources.test.images, datasources.test.caption_groups)
        
        pregen_time = sub_timer.get_duration()
        print(' time:', lib.format_duration(pregen_time))
        print(' score:', pregen_score)
        
        print(dataset_name, architecture, run, pregen_time, file=f)
        print('', file=f)
        
        ########################################################################################
        print('measuring postgen timing')
        print('postgen', file=f)
        print('dataset', 'architecture', 'run', 'beam_width', 'lower_bound_len', 'upper_bound_len', 'postgen_time', file=f)
        
        for upper_bound_len in [ 50, 5]:
            print(' len:', upper_bound_len)
            sub_timer = lib.Timer()
            
            captions_tokens = m.generate_captions_beamsearch(datasources.test.images, beam_width=beam_width, lower_bound_len=lower_bound_len, upper_bound_len=upper_bound_len, temperature=1.0)
            
            postgen_time = sub_timer.get_duration()
            print(' time:', lib.format_duration(postgen_time))
            
            print(dataset_name, architecture, run, beam_width, lower_bound_len, upper_bound_len, postgen_time, file=f)