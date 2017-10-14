import tensorflow as tf
import numpy as np
import glob  # this will be useful when reading reviews from file
import os
import tarfile
import re

batch_size = 50


def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""

    data = []
    reviewlist = []
    # extract files
    filename = "reviews.tar.gz"
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data/')):
        with tarfile.open(filename, "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data/'))
    dir = os.path.dirname(__file__)
    # read files
    reviews = glob.glob(os.path.join(dir, 'data/pos/*'))
    reviews.extend(glob.glob(os.path.join(dir, 'data/neg/*')))
    for rev in reviews:
        with open(rev, "r", encoding='utf-8') as r:
            line = r.readline()
            sentence = [i.lower() for i in re.findall(r'[\w]+', line)]
            reviewlist.append(sentence[:40])
    # convert word to index
    for review in reviewlist:
        sen = []
        for word in review:
            if word in glove_dict:
                sen.append(glove_dict[word])
            else:
                sen.append(0)
        while (len(sen) < 40):
            sen.append(0)
        data.append(sen)
    data = np.array(data, dtype=np.int32)
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    # data = open("glove.6B.50d.txt",'r',encoding="utf-8")
    # if you are running on the CSE machines, you can load the glove data from here
    # data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    data = open("glove.6B.50d.txt", 'r', encoding="utf-8")
    word_index_dict = {}
    word_index_dict['UNK'] = 0
    embeddings = []
    embeddings.append(50 * [0])
    index = 1
    for line in data:
        sline = line.split()
        word_index_dict[sline[0]] = index
        embeddings.append([float(i) for i in sline[1:]])
        index = index + 1
    embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    lstm_num = 16
    num_classes = 2
    seq_length = 40

    # input and output
    labels = tf.placeholder(tf.float32, [batch_size, num_classes], name="labels")
    input_data = tf.placeholder(tf.int32, [batch_size, seq_length], name="input_data")
    # looking up from np array
    data = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)
    # create lstm cell and apply drop out
    _lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_num, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=_lstm_cell, output_keep_prob=0.75)
    # use dynamic rnn
    rnn_out, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)
    W = tf.Variable(tf.truncated_normal([lstm_num, num_classes]))
    b = tf.Variable(tf.constant(0.05, shape=[num_classes]))
    # transform rnn_out from (50,40,16) to (40,50,16)
    trans_rnn = tf.transpose(rnn_out, [1, 0, 2])
    # slice the last one from tans_rnn
    last_elem = trans_rnn[-1]
    logits = (tf.matmul(last_elem, W) + b)
    # make prediction and calculate the loss
    prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name="accuracy")
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(losses, name="loss")
    # optimize with learning decay
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.2, global_step, 5000, 0.9, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
