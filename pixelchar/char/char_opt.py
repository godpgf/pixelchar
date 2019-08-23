import tensorflow as tf
from pixelchar.data import *


def length(coff_list):
    return len(coff_list[0][1])


def matrix(coff_list):
    stddev = 0.02 if len(coff_list) == 1 else coff_list[1][1]
    return tf.Variable(tf.truncated_normal(shape=[int(coff) for coff in coff_list[0][1]], stddev=stddev))


def dropout(coff_list):
    data = coff_list[0][1]
    coff = coff_list[1][1]
    return tf.nn.dropout(data, coff)


def array(coff_list):
    return [coff[1] for coff in coff_list]


def get_value(coff_list):
    value_array = coff_list[0][1]
    return value_array[int(coff_list[1][1])]


def set_value(coff_list):
    return (coff_list[0][1],
            coff_list[0][2])


def reduce_sum(coff_list):
    if isinstance(coff_list[0][1], list):
        return [tf.reduce_sum(a, axis=int(coff_list[1][1]) if len(coff_list) > 1 else 0) for a in coff_list[0][1]]
    else:
        return tf.reduce_sum(coff_list[0][1], axis=int(coff_list[1][1]) if len(coff_list) > 1 else 0)


def reduce_mean(coff_list):
    if isinstance(coff_list[0][1], list):
        return [tf.reduce_mean(a, axis=int(coff_list[1][1]) if len(coff_list) > 1 else 0) for a in coff_list[0][1]]
    else:
        return tf.reduce_mean(coff_list[0][1], axis=int(coff_list[1][1]) if len(coff_list) > 1 else 0)


def embedding_lookup(coff_list):
    if isinstance(coff_list[0][1], list):
        return [tf.nn.embedding_lookup(a, b) for a, b in zip(coff_list[0][1], coff_list[1][1])]
    else:
        return tf.nn.embedding_lookup(coff_list[0][1], coff_list[1][1])


def ffm_embedding_lookup(coff_list):
    fid = 0
    res_list = []
    for i in range(len(coff_list[1][1]) - 1):
        for j in range(i + 1, len(coff_list[1][1])):
            a = coff_list[0][1][fid]
            b = coff_list[1][1][i]
            res_list.append(tf.nn.embedding_lookup(a, b))
            fid += 1

            a = coff_list[0][1][fid]
            b = coff_list[1][1][j]
            res_list.append(tf.nn.embedding_lookup(a, b))
            fid += 1
    return res_list


def reshape(coff_list):
    return tf.reshape(coff_list[0][1], shape=[int(coff) for coff in coff_list[1][1]])


def adam_optim(coff_list):
    lr = coff_list[0][1]
    return tf.train.AdamOptimizer(lr, name='Adam')


def ftrl_optim(coff_list):
    lr = coff_list[0][1]
    return tf.train.FtrlOptimizer(lr, name='Ftrl')


def minimize(coff_list):
    optim = coff_list[0][1]
    value = coff_list[1][1]
    return optim.minimize(value)


def _create_mat(data_meta, embed_size, stddev):
    if isinstance(data_meta, BarDataMeta):
        return tf.Variable(
            tf.truncated_normal(shape=[data_meta.width * data_meta.height, embed_size], stddev=stddev))
    elif isinstance(data_meta, SeqDataMeta):
        return tf.Variable(
            tf.truncated_normal(shape=[data_meta.max_item_size, embed_size], stddev=stddev))


def embed_matrix(data_meta_dict, coff_list):
    embed_size = int(coff_list[1][1])
    stddev = coff_list[2][1] if len(coff_list) > 2 else 0.02
    if isinstance(coff_list[0][1], list):
        name_list = coff_list[0][0].split(':')[1].split(',')
        data_meta_list = [data_meta_dict[name] for name in name_list]
        matrix_list = []
        for data_meta in data_meta_list:
            matrix_list.append(_create_mat(data_meta, embed_size, stddev))
        return matrix_list
    else:
        return _create_mat(data_meta_dict[coff_list[0][0]], embed_size, stddev)


def ffm_embed_matrix(data_meta_dict, coff_list):
    embed_size = int(coff_list[1][1])
    stddev = coff_list[2][1] if len(coff_list) > 2 else 0.02
    name_list = coff_list[0][0].split(':')[1].split(',')
    data_meta_list = [data_meta_dict[name] for name in name_list]
    matrix_list = []
    for i in range(0, len(data_meta_list) - 1):
        for j in range(i+1, len(data_meta_list)):
            matrix_list.append(_create_mat(data_meta_list[i], embed_size, stddev))
            matrix_list.append(_create_mat(data_meta_list[j], embed_size, stddev))
    return matrix_list


def l2_loss(coff_list):
    v = coff_list[0][1]
    if isinstance(v, list):
        r = tf.nn.l2_loss(v[0])
        for i in range(1, len(v)):
            r = r + tf.nn.l2_loss(v[i])
        return r
    else:
        return tf.nn.l2_loss(v)


def l1_loss(coff_list):
    v = coff_list[0][1]
    if isinstance(v, list):
        r = tf.reduce_mean(tf.abs(tf.reshape(v[0], [-1])), 0)
        for i in range(1, len(v)):
            r = r + tf.reduce_mean(tf.abs(tf.reshape(v[i], [-1])), 0)
        return r
    else:
        return tf.reduce_mean(tf.abs(tf.reshape(v, [-1])), 0)


def add_n(coff_list):
    c_list = [v for v in coff_list[0][1]]
    return tf.add_n(c_list)


def dot(coff_list):
    v0 = coff_list[0][1]
    v1 = coff_list[1][1]
    if isinstance(v0, list) and isinstance(v1, list):
        r = tf.reshape(tf.reduce_sum(v0[0] * v1[0], axis=1), shape=[-1, 1])
        for i in range(1, len(v0)):
            r = r + tf.reshape(tf.reduce_sum(v0[i] * v1[i], axis=1), shape=[-1, 1])
        return r
    else:
        return tf.reshape(tf.reduce_sum(v0 * v1, axis=1), shape=[-1, 1])


def concat(coff_list):
    tf_list = [c for c in coff_list[0][1]]
    return tf.concat(tf_list, axis=int(coff_list[1][1]))


def matmul(coff_list):
    return tf.matmul(coff_list[0][1], coff_list[1][1])


def sigmoid(coff_list):
    return tf.nn.sigmoid(coff_list[0][1])


def softmax(coff_list):
    return tf.nn.softmax(coff_list[0][1])


def relu(coff_list):
    return tf.nn.relu(coff_list[0][1])


def tanh(coff_list):
    return tf.nn.tanh(coff_list[0][1])


def kernel_fm(coff_list):
    tf_list = [c for c in coff_list[0][1]]
    kernel_mat = coff_list[1][1]
    res_list = []
    pid = 0
    for i in range(0, len(tf_list) - 1):
        for j in range(i + 1, len(tf_list)):
            k_mat = tf.gather(kernel_mat, pid)
            res_list.append(tf.reshape(tf.reduce_sum(tf.matmul(tf_list[i], k_mat) * tf_list[j], axis=1), shape=[-1, 1]))
    return res_list


def fm(coff_list):
    tf_list = [c for c in coff_list[0][1]]
    res_list = []
    for i in range(0, len(tf_list) - 1):
        for j in range(i + 1, len(tf_list)):
            res_list.append(tf.reshape(tf.reduce_sum(tf_list[i] * tf_list[j], axis=1), shape=[-1, 1]))
    return res_list


def ffm(coff_list):
    tf_list = [c for c in coff_list[0][1]]
    res_list = []
    for i in range(0, len(tf_list), 2):
        res_list.append(tf.reshape(tf.reduce_sum(tf_list[i] * tf_list[i + 1], axis=1), shape=[-1, 1]))
    return res_list


def sigmoid_cross_entropy_with_logits(coff_list):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=coff_list[0][1], logits=coff_list[1][1])


def cross_entropy_with_logits(coff_list):
    # 注意没有sigmoid
    z = coff_list[0][1]
    x = coff_list[1][1]
    return z * -tf.log(x) + (1 - z) * -tf.log(1 - x)
