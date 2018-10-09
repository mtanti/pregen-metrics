import numpy as np
import json

import lib
import model_base
import model_normal
import data
import helper_datasources
import config

########################################################################################
def get_probs(model, images, caption_groups):
    probs = []
    for (img, caption_group) in zip(images, caption_groups):
        probs.append([])
        for cap in caption_group:
            probs[-1].append([])
            cap_len = len(cap) + 1
            cap_prefix = [data.EDGE_INDEX] + [ model.dataset.token_to_index.get(token, data.UNKNOWN_INDEX) for token in cap ]
            cap_target = [ model.dataset.token_to_index.get(token, data.UNKNOWN_INDEX) for token in cap ] + [data.EDGE_INDEX]
            distributions = model.get_raw_probs([img], [cap_prefix], [cap_len], 1.0)
            for (distribution, target) in zip(distributions[0], cap_target):
                sorted_indexes = np.argsort(-distribution).tolist()
                probs[-1][-1].append((distribution[target].tolist(), sorted_indexes.index(target)))
    return probs
    
########################################################################################
for dataset_name in [ 'flickr8k', 'flickr30k', 'mscoco' ]:
    datasources = helper_datasources.DataSources(dataset_name)
    dataset = data.Dataset(
            min_token_freq        = config.min_token_freq,
            training_datasource   = datasources.train,
            validation_datasource = datasources.val,
            testing_datasource    = datasources.test,
        )
    dataset.process()
    
    for architecture in [ 'init', 'pre', 'par', 'merge' ]:
        for run in range(1, config.num_runs+1):
            print('{}_{}_{}'.format(dataset_name, architecture, run))
            
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
                test_grouped_probs = get_probs(m, datasources.test.images, datasources.test.caption_groups)
            
            with open('test_probs/{}_{}_{}.json'.format(dataset_name, architecture, run), 'w', encoding='utf-8') as f:
                json.dump(test_grouped_probs, f)
            
