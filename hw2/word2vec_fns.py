import tensorflow as tf
import numpy as np
import collections

data_index = 0

def generate_batch(data, batch_size, skip_window):
    """
    Generates a mini-batch of training data for the training CBOW
    embedding model.
    :param data (numpy.ndarray(dtype=int, shape=(corpus_size,)): holds the
        training corpus, with words encoded as an integer
    :param batch_size (int): size of the batch to generate
    :param skip_window (int): number of words to both left and right that form
        the context window for the target word.
    Batch is a vector of shape (batch_size, 2*skip_window), with each entry for the batch containing all the context words, with the corresponding label being the word in the middle of the context
    """
    global data_index
    ## assert batch_size % num_skips == 0
    ## assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size, skip_window * 2), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    #get words starting from index 0 to span
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size):
        target = skip_window  # target label at the center of the buffer
        batch[i] = [buffer[i] for i in range(len(buffer)) if not i == skip_window]
        labels[i] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

def get_mean_context_embeds(embeddings, train_inputs):
    """
    :param embeddings (tf.Variable(shape=(vocabulary_size, embedding_size))
    :param train_inputs (tf.placeholder(shape=(batch_size, 2*skip_window))
    returns:
        `mean_context_embeds`: the mean of the embeddings for all context words
        for each entry in the batch, should have shape (batch_size,
        embedding_size)
    """
    # cpu is recommended to avoid out of memory errors, if you don't
    # have a high capacity GPU
    with tf.device('/cpu:0'):
        tmp = tf.nn.embedding_lookup(embeddings, train_inputs)
        mean_context_embeds = tf.reduce_mean(tmp , 1)
    return mean_context_embeds
