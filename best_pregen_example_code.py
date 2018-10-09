#For a faster, batched version of the below algorithm, see get_timings.py. This is the more explained and shorter version.

import numpy as np

def pregen(images, refcaption_groups_indexes, session, caption_placeholder, image_placeholder, softmax):
    '''
    Measure the predicted caption quality of a trained caption generator model using the best found pregen metric.
    This function assumes that the reference captions are grouped into separate lists for each corresponding image (since there is more than one reference caption per image) and that the captions have been converted into indexes that correspond to the softmax indexes. It also assumes that each caption begins with the start token index and ends with the end token index.
        images:
            [
                img1,
                img2,
                img3,
                ...
            ]
        refcaption_groups_indexes:
            [
                [
                    [ start_index, img1_cap1_word1, img1_cap1_word2, img1_cap1_word3, ..., end_index ],
                    [ start_index, img1_cap2_word1, img1_cap2_word2, img1_cap2_word3, ..., end_index ],
                    [ start_index, img1_cap3_word1, img1_cap3_word2, img1_cap3_word3, ..., end_index ],
                    ...
                ],
                [
                    [ start_index, img2_cap1_word1, img2_cap1_word2, img2_cap1_word3, ..., end_index ],
                    [ start_index, img2_cap2_word1, img2_cap2_word2, img2_cap2_word3, ..., end_index ],
                    [ start_index, img2_cap3_word1, img2_cap3_word2, img2_cap3_word3, ..., end_index ],
                    ...
                ],
                [
                    [ start_index, img3_cap1_word1, img3_cap1_word2, img3_cap1_word3, ..., end_index ],
                    [ start_index, img3_cap2_word1, img3_cap2_word2, img3_cap2_word3, ..., end_index ],
                    [ start_index, img3_cap3_word1, img3_cap3_word2, img3_cap3_word3, ..., end_index ],
                    ...
                ],
                ...
            ]
        softmax:
            Softmax is run in a session with caption_placeholder containing a single vector representing a single sentence of indexes: [ start_index, word1, word2, ... ]
            When run in a session, softmax should return a matrix of probabilities. Each row in the matrix should be a distribution of probabilities for the next word in a prefix of the sentence:
            [
                [ prob_word1_vocab1, prob_word1_vocab2, prob_word1_vocab3, ... ],
                [ prob_word2_vocab1, prob_word2_vocab2, prob_word2_vocab3, ... ],
                [ prob_word3_vocab1, prob_word3_vocab2, prob_word3_vocab3, ... ],
                ...
            ]
            where prob_word1_vocab1 is the probability of the first word in the vocabulary (having index 0) following the start token (being the first word), prob_word1_vocab2 is the probability of the second word in the vocabulary (having index 1) following the start token (being the first word), prob_word2_vocab1 is the probability of the first word in the vocabulary (having index 0) following the start token and word1, etc.
    '''
    img_scores = []
    for (img, caption_group) in zip(images, refcaption_groups_indexes):
        cap_scores = []
        for cap in caption_group:
            distributions = session.run(softmax, { caption_placeholder: cap[:-1], image_placeholder: img }) #Get the distribution of word probabilities for every word position in the caption
            maxprobs_prefix_len = 0
            for (distribution, target) in zip(distributions, cap[1:]): #Get the word probabilities and actual next word for the next word position
                if np.argmax(distribution) == target: #If the maximum probability word is the correct next word then increase the prefix length
                    maxprobs_prefix_len += 1
                else:
                    break
            score = maxprobs_prefix_len/(len(cap)-1) #Normalise the prefix length to get the caption score
            cap_scores.append(score)
        img_scores.append(max(cap_scores)) #Find the maximum caption score to get the image score
    return np.mean(img_scores) #Find the mean image score to get the dataset score