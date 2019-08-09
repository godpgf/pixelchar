import tensorflow as tf
from pixelchar.data import *


def matrix(coff_list):
    stddev = 0.02 if len(coff_list) == 1 else coff_list[1][1]
    return tf.Variable(tf.truncated_normal(shape=[int(coff) for coff in coff_list[0][1]], stddev=stddev))


def array(coff_list):
    return [coff[1] for coff in coff_list]


def reduce_sum(coff_list):
    return tf.reduce_sum(coff_list[0][1], axis=int(coff_list[1][1]) if len(coff_list) > 1 else 0)


def reduce_mean(coff_list):
    return tf.reduce_mean(coff_list[0][1], axis=int(coff_list[1][1]) if len(coff_list) > 1 else 0)


def embedding_lookup(coff_list):
    return tf.nn.embedding_lookup(coff_list[0][1], coff_list[1][1])


def reshape(coff_list):
    return tf.reshape(coff_list[0][1], shape=[int(coff) for coff in coff_list[1][1]])


def adam_optim(coff_list):
    lr = coff_list[0][1]
    return tf.train.AdamOptimizer(lr, name='Adam')


def minimize(coff_list):
    optim = coff_list[0][1]
    value = coff_list[1][1]
    return optim.minimize(value)


def embed_matrix(data_meta_dict, coff_list):
    embed_size = int(coff_list[1][1])
    stddev = coff_list[2][1] if len(coff_list) > 2 else 0.02
    data_meta = data_meta_dict[coff_list[0][0]]
    if isinstance(data_meta, BarDataMeta):
        return tf.Variable(
            tf.truncated_normal(shape=[data_meta.width * data_meta.height, embed_size], stddev=stddev))
    elif isinstance(data_meta, SeqDataMeta):
        return tf.Variable(
            tf.truncated_normal(shape=[data_meta.max_item_size, embed_size], stddev=stddev))


def l2_loss(coff_list):
    return tf.nn.l2_loss(coff_list[0][1])


def dot(coff_list):
    v0 = coff_list[0][1]
    v1 = coff_list[1][1]
    return tf.reshape(tf.reduce_sum(v0 * v1, axis=1), shape=[-1, 1])


def concat(coff_list):
    tf_list = [c for c in coff_list[0][1]]
    return tf.concat(tf_list, axis=int(coff_list[1][1]))


def matmul(coff_list):
    return tf.matmul(coff_list[0][1], coff_list[1][1])


def sigmoid(coff_list):
    return tf.nn.sigmoid(coff_list[0][1])


def sigmoid_cross_entropy_with_logits(coff_list):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=coff_list[0][1], logits=coff_list[1][1])
