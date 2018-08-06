'''
derived from Text2Shape by Kevin Chen, see: https://github.com/kchen92/text2shape
'''

import argparse
import collections
import datetime
import json
import numpy as np
import os
import pickle
import sys


def construct_embeddings_matrix(dataset, embedding, mode):
    """Construct the embeddings matrix, which is NxD where N is the number of embeddings and D is
    the dimensionality of each embedding.

    Args:
        dataset: String specifying the dataset
        embedding: Dictionary containing the embeddings. 
            form: emb = {
                'shape_embedding': shape_emb,
                'text_embedding': text_emb
            }
    """
    
    num_sample = len(embedding)
    num_shape = num_sample
    num_text = np.sum([1 for key in embedding.keys() for _ in embedding[key]['text_embedding']])
    embedding_dim = embedding[list(embedding.keys())[0]]['shape_embedding'].shape[0]

    # Print info about embeddings
    print('\nNumber of embedding:', num_sample)
    print('Number of shape embeddings: {}, number of text embeddings: {}'.format(num_shape, num_text))
    print('Dimensionality of embedding:', embedding_dim)
    print()

    # extract embedding
    shape_embedding = [(key, embedding[key]['shape_embedding']) for key in embedding.keys()]
    text_embedding = [(key, item) for key in embedding.keys() for item in embedding[key]['text_embedding']]

    # process shape embedding
    shape_matrix = np.zeros((num_shape, embedding_dim))
    shape_labels = np.zeros((num_shape)).astype(int)
    shape_idx2label, shape_label2idx = {}, {}
    label_counter = 0
    for idx, data in enumerate(shape_embedding):
        # Parse caption tuple
        model_id, emb = data

        # # Swap model ID and category depending on dataset
        # if dataset == 'primitives':
        #     tmp = model_id
        #     model_id = category
        #     category = tmp

        # Add model ID to dict if it has not already been added
        if model_id not in shape_idx2label:
            shape_idx2label[model_id] = label_counter
            shape_label2idx[label_counter] = model_id
            label_counter += 1

        # Update the embeddings matrix and labels vector
        shape_matrix[idx] = emb
        shape_labels[idx] = shape_idx2label[model_id]
    
    # process text embedding
    text_matrix = np.zeros((num_text, embedding_dim))
    text_labels = np.zeros((num_text)).astype(int)
    text_idx2label, text_label2idx = {}, {}
    label_counter = 0
    for idx, data in enumerate(text_embedding):
        # Parse caption tuple
        model_id, emb = data

        # # Swap model ID and category depending on dataset
        # if dataset == 'primitives':
        #     tmp = model_id
        #     model_id = category
        #     category = tmp

        # Add model ID to dict if it has not already been added
        if model_id not in text_idx2label:
            text_idx2label[model_id] = label_counter
            text_label2idx[label_counter] = model_id
            label_counter += 1

        # Update the embeddings matrix and labels vector
        text_matrix[idx] = emb[1]
        text_labels[idx] = text_idx2label[model_id]

    if mode == 't2t':
        query_embedding = text_matrix
        target_embedding = text_matrix
        labels = text_labels
        target_labels = text_labels
        idx2label = text_idx2label
        label2idx = text_label2idx
    elif mode == 's2t':
        query_embedding = shape_matrix
        target_embedding = text_matrix
        labels = shape_labels
        target_labels = [shape_idx2label[item[0]] for item in text_embedding]
        idx2label = shape_idx2label
        label2idx = shape_label2idx
    elif mode == 't2s':
        query_embedding = text_matrix
        target_embedding = shape_matrix
        labels = text_labels
        target_labels = [text_idx2label[item[0]] for item in shape_embedding]
        idx2label = text_idx2label
        label2idx = text_label2idx
    else:
        raise ValueError("unsupported mode, please choose from t2t/t2s/s2t")
    
    return query_embedding, target_embedding, labels, target_labels, idx2label, label2idx, 


def print_model_id_info(model_id_to_label):
    print('Number of models (or categories if synthetic dataset):', len(model_id_to_label.keys()))
    print('')

    # Look at a few example model IDs
    print('Example model IDs:')
    for i, k in enumerate(model_id_to_label):
        if i < 10:
            print(k)
    print('')


def _compute_nearest_neighbors_cosine(target_embeddings_matrix, query_embeddings_matrix,
                                      n_neighbors, fit_eq_query, range_start=0):

    n_neighbors += 1

    # print('Using unnormalized cosine distance')

    # Argsort method
    # unnormalized_similarities = np.dot(query_embeddings_matrix, target_embeddings_matrix.T)
    # sort_indices = np.argsort(unnormalized_similarities, axis=1)
    # # return unnormalized_similarities[:, -n_neighbors:], sort_indices[:, -n_neighbors:]
    # indices = sort_indices[:, -n_neighbors:]
    # indices = np.flip(indices, 1)

    # Argpartition method
    unnormalized_similarities = np.dot(query_embeddings_matrix, target_embeddings_matrix.T)
    n_samples = unnormalized_similarities.shape[0]
    sort_indices = np.argpartition(unnormalized_similarities, -n_neighbors, axis=1)
    indices = sort_indices[:, -n_neighbors:]
    row_indices = [x for x in range(n_samples) for _ in range(n_neighbors)]
    yo = unnormalized_similarities[row_indices, indices.flatten()].reshape(n_samples, n_neighbors)
    indices = indices[row_indices, np.argsort(yo, axis=1).flatten()].reshape(n_samples, n_neighbors)
    indices = np.flip(indices, 1)

    n_neighbors -= 1  # Undo the neighbor increment

    final_indices = np.zeros((indices.shape[0], n_neighbors), dtype=int)
    compare_mat = np.asarray(list(range(range_start, range_start + indices.shape[0]))).reshape(indices.shape[0], 1)
    has_self = np.equal(compare_mat, indices)  # has self as nearest neighbor
    any_result = np.any(has_self, axis=1)
    for row_idx in range(indices.shape[0]):
        if any_result[row_idx]:
            nonzero_idx = np.nonzero(has_self[row_idx, :])
            assert len(nonzero_idx) == 1
            new_row = np.delete(indices[row_idx, :], nonzero_idx[0])
            final_indices[row_idx, :] = new_row
        else:
            final_indices[row_idx, :] = indices[row_idx, :n_neighbors]
    indices = final_indices
    
    return indices


def compute_nearest_neighbors_cosine(target_embeddings_matrix, query_embeddings_matrix,
                                     n_neighbors, fit_eq_query):
    print('Using unnormalized cosine distance')
    n_samples = query_embeddings_matrix.shape[0]
    if n_samples > 8000:  # Divide into blocks and execute
        def block_generator(mat, block_size):
            for i in range(0, mat.shape[0], block_size):
                yield mat[i:(i + block_size), :]

        block_size = 3000
        blocks = block_generator(query_embeddings_matrix, block_size)
        indices_list = []
        for cur_block_idx, block in enumerate(blocks):
            print('Nearest neighbors on block {}'.format(cur_block_idx + 1))
            cur_indices = _compute_nearest_neighbors_cosine(target_embeddings_matrix, block,
                                                            n_neighbors, fit_eq_query,
                                                            range_start=cur_block_idx * block_size)
            indices_list.append(cur_indices)
        indices = np.vstack(indices_list)
        return None, indices
    else:
        return None, _compute_nearest_neighbors_cosine(target_embeddings_matrix,
                                                       query_embeddings_matrix, n_neighbors,
                                                       fit_eq_query)


def compute_nearest_neighbors(target_embeddings_matrix, query_embeddings_matrix,
                              n_neighbors, metric='minkowski'):
    """Compute nearest neighbors.

    Args:
        target_embeddings_matrix: NxD matrix
    """
    fit_eq_query = False
    if ((target_embeddings_matrix.shape == query_embeddings_matrix.shape)
        and np.allclose(target_embeddings_matrix, query_embeddings_matrix)):
        fit_eq_query = True

    if metric == 'cosine':
        distances, indices = compute_nearest_neighbors_cosine(target_embeddings_matrix,
                                                              query_embeddings_matrix,
                                                              n_neighbors, fit_eq_query)
    else:
        raise ValueError('Use cosine distance.')
    return distances, indices


def compute_pr_at_k(indices, labels, n_neighbors, num_embeddings, fit_labels=None):
    """Compute precision and recall at k (for k=1 to n_neighbors)

    Args:
        indices: num_embeddings x n_neighbors array with ith entry holding nearest neighbors of
                 query i
        labels: 1-d array with correct class of query
        n_neighbors: number of neighbors to consider
        num_embeddings: number of queries
    """
    if fit_labels is None:
        fit_labels = labels
    num_correct = np.zeros((num_embeddings, n_neighbors))
    rel_score = np.zeros((num_embeddings, n_neighbors))
    label_counter = np.bincount(fit_labels)
    num_relevant = label_counter[labels]
    rel_score_ideal = np.zeros((num_embeddings, n_neighbors))

    # Assumes that self is not included in the nearest neighbors
    for i in range(num_embeddings):
        label = labels[i]  # Correct class of the query
        nearest = indices[i]  # Indices of nearest neighbors
        nearest_classes = [fit_labels[x] for x in nearest]  # Class labels of the nearest neighbors
        # for now binary relevance
        num_relevant_clamped = min(num_relevant[i], n_neighbors)
        rel_score[i] = np.equal(np.asarray(nearest_classes), label)
        rel_score_ideal[i][0:num_relevant_clamped] = 1

        for k in range(n_neighbors):
            # k goes from 0 to n_neighbors-1
            correct_indicator = np.equal(np.asarray(nearest_classes[0:(k + 1)]), label)  # Get true (binary) labels
            num_correct[i, k] = np.sum(correct_indicator)

    # Compute our dcg
    dcg_n = np.exp2(rel_score) - 1
    dcg_d = np.log2(np.arange(1,n_neighbors+1)+1)
    dcg = np.cumsum(dcg_n/dcg_d,axis=1)
    # Compute ideal dcg
    dcg_n_ideal = np.exp2(rel_score_ideal) - 1
    dcg_ideal = np.cumsum(dcg_n_ideal/dcg_d,axis=1)
    # Compute ndcg
    ndcg = dcg / dcg_ideal
    ave_ndcg_at_k = np.sum(ndcg, axis=0) / num_embeddings
    recall_rate_at_k = np.sum(num_correct > 0, axis=0) / num_embeddings
    recall_at_k = np.sum(num_correct/num_relevant[:,None], axis=0) / num_embeddings
    precision_at_k = np.sum(num_correct/np.arange(1,n_neighbors+1), axis=0) / num_embeddings
    #print('recall_at_k shape:', recall_at_k.shape)
    print('     k: precision recall recall_rate ndcg')
    for k in range(n_neighbors):
        print('pr @ {}: {:.5f} {:.5f} {:.5f} {:.5f}'.format(k + 1, precision_at_k[k], recall_at_k[k], recall_rate_at_k[k], ave_ndcg_at_k[k]))
    Metrics = collections.namedtuple('Metrics', 'precision recall recall_rate ndcg')
    return Metrics(precision_at_k, recall_at_k, recall_rate_at_k, ave_ndcg_at_k)


def get_nearest_info(indices, labels, label_to_model_id, caption_tuples, idx_to_word):
    """Compute and return the model IDs of the nearest neighbors.
    """
    # Convert labels to model IDs
    query_model_ids = []
    query_sentences = []
    for idx, label in enumerate(labels):
        query_model_ids.append(caption_tuples[idx][2])
        cur_sentence_as_word_indices = caption_tuples[idx][0]
        if cur_sentence_as_word_indices is None:
            query_sentences.append('None (shape embedding)')
        else:
            query_sentences.append('None (text embedding)')

    # Convert neighbors to model IDs
    nearest_model_ids = []
    nearest_sentences = []
    for row in indices:
        model_ids = []
        sentences = []
        for col in row:
            model_ids.append(caption_tuples[col][2])
            cur_sentence_as_word_indices = caption_tuples[col][0]
            if cur_sentence_as_word_indices is None:
                cur_sentence_as_words = 'None (shape embedding)'
            else:
                cur_sentence_as_words = 'None (text embedding)'
            sentences.append(cur_sentence_as_words)
        nearest_model_ids.append(model_ids)
        nearest_sentences.append(sentences)
    assert len(query_model_ids) == len(nearest_model_ids)
    return query_model_ids, nearest_model_ids, query_sentences, nearest_sentences


def print_nearest_info(query_model_ids, nearest_model_ids, query_sentences, nearest_sentences):
    """Print out nearest model IDs for random queries.

    Args:
        labels: 1D array containing the label
    """
    pass


def compute_metrics(dataset, embeddings_dict, mode, metric='minkowski'):
    """Compute all the metrics for the text encoder evaluation.
    """
    (query_embeddings_matrix, target_embeddings_matrix, labels, target_labels, model_id_to_label, label_to_model_id) = construct_embeddings_matrix(
        dataset,
        embeddings_dict,
        mode
    )
    print_model_id_info(model_id_to_label)

    n_neighbors = 20

    distances, indices = compute_nearest_neighbors(target_embeddings_matrix, query_embeddings_matrix, n_neighbors, metric=metric)

    print('Computing precision recall.')
    pr_at_k = compute_pr_at_k(indices, labels, n_neighbors, query_embeddings_matrix.shape[0], fit_labels=target_labels)

    # query_model_ids, nearest_model_ids, query_sentences, nearest_sentences = get_nearest_info(
    #     indices,
    #     labels,
    #     label_to_model_id,
    #     embeddings_dict['caption_embedding_tuples'],
    #     None
    # )

    return pr_at_k


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help='dataset (''shapenet'', ''primitives'')')
    parser.add_argument('--embedding', help='path to the root folder containing embeddings')
    parser.add_argument('--phase', help='train/val/test')
    parser.add_argument('--mode', help='t2t/t2s/s2t', type=str)
    args = parser.parse_args()

    with open("outputs/embedding/{}/embeddings/{}.p".format(args.embedding, args.phase), 'rb') as f:
        embedding = pickle.load(f)

    np.random.seed(1234)
    compute_metrics(args.dataset, embedding, mode=args.mode, metric='cosine')


if __name__ == '__main__':
    main()