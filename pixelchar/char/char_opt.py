import tensorflow as tf
import numpy as np
import math


def _get_max_item_size(name):
    return int(name.split('_')[-1].split(':')[0])


def length(coff_list):
    return len(coff_list[0])


def matrix(coff_list):
    stddev = 0.02 if len(coff_list) == 1 else coff_list[1]
    return tf.Variable(tf.truncated_normal(shape=[int(coff) for coff in coff_list[0]], stddev=stddev))


def dropout(coff_list):
    data = coff_list[0]
    coff = coff_list[1]
    return tf.nn.dropout(data, coff)


def array(coff_list):
    return [coff for coff in coff_list]


def get_value(coff_list):
    value_array = coff_list[0]
    return value_array[int(coff_list[1])]


def set_value(coff_list):
    return (coff_list[0],
            int(coff_list[1]))


def reduce_sum(coff_list):
    if isinstance(coff_list[0], list):
        return [tf.reduce_sum(a, axis=int(coff_list[1]) if len(coff_list) > 1 else 0) for a in coff_list[0]]
    else:
        return tf.reduce_sum(coff_list[0], axis=int(coff_list[1]) if len(coff_list) > 1 else 0)


def reduce_mean(coff_list):
    if isinstance(coff_list[0], list):
        return [tf.reduce_mean(a, axis=int(coff_list[1]) if len(coff_list) > 1 else 0) for a in coff_list[0]]
    else:
        return tf.reduce_mean(coff_list[0], axis=int(coff_list[1]) if len(coff_list) > 1 else 0)


def embedding_lookup(coff_list):
    if isinstance(coff_list[0], list):
        return [tf.nn.embedding_lookup(a, b) for a, b in zip(coff_list[0], coff_list[1])]
    else:
        return tf.nn.embedding_lookup(coff_list[0], coff_list[1])


def ffm_embedding_lookup(coff_list):
    fid = 0
    res_list = []
    for i in range(len(coff_list[1]) - 1):
        for j in range(i + 1, len(coff_list[1])):
            a = coff_list[0][fid]
            b = coff_list[1][i]
            res_list.append(tf.nn.embedding_lookup(a, b))
            fid += 1

            a = coff_list[0][fid]
            b = coff_list[1][j]
            res_list.append(tf.nn.embedding_lookup(a, b))
            fid += 1
    return res_list


def reshape(coff_list):
    return tf.reshape(coff_list[0], shape=[int(coff) for coff in coff_list[1]])


def adam_optim(coff_list):
    lr = coff_list[0]
    return tf.train.AdamOptimizer(lr, name='Adam')


def grad_optim(coff_list):
    lr = coff_list[0]
    return tf.train.GradientDescentOptimizer(lr, name='Grad')


def ftrl_optim(coff_list):
    lr = coff_list[0]
    return tf.train.FtrlOptimizer(lr, name='Ftrl')


def minimize(coff_list):
    optim = coff_list[0]
    value = coff_list[1]
    return optim.minimize(value)


def log(coff_list):
    if isinstance(coff_list[0], list):
        return [tf.log(c) for c in coff_list[0]]
    else:
        return tf.log(coff_list[0])


def one_hot(coff_list):
    if isinstance(coff_list[0], list):
        res = []
        for v in coff_list[0]:
            max_item_size = _get_max_item_size(v.name)
            res.append(tf.one_hot(v, max_item_size, 1.0, 0.0))
    else:
        max_item_size = _get_max_item_size(coff_list[0].name)
        return tf.one_hot(coff_list[0], max_item_size, 1.0, 0.0)


def variable(coff_list):
    value = float(coff_list[1])
    if isinstance(coff_list[0], list):
        v_list = []
        for v in coff_list[0]:
            v_list.append(tf.Variable(tf.zeros([_get_max_item_size(v.name)]) + value))
        return v_list
    else:
        return tf.Variable(tf.zeros([_get_max_item_size(coff_list[0].name)]) + value)


def _create_mat(value, embed_size, stddev):
    max_item_size = _get_max_item_size(value.name)
    return tf.Variable(
        tf.truncated_normal(shape=[max_item_size, embed_size], stddev=stddev))


def embed_matrix(coff_list):
    embed_size = int(coff_list[1])
    stddev = coff_list[2] if len(coff_list) > 2 else 1.0 / math.sqrt(embed_size)
    if isinstance(coff_list[0], list):
        matrix_list = []
        for v in coff_list[0]:
            matrix_list.append(_create_mat(v, embed_size, stddev))
        return matrix_list
    else:
        return _create_mat(coff_list[0], embed_size, stddev)


def _create_uniform_mat(value, embed_size):
    max_item_size = _get_max_item_size(value.name)
    return tf.Variable(tf.random_uniform([max_item_size, embed_size], -1.0, 1.0))


def embed_uniform_matrix(coff_list):
    embed_size = int(coff_list[1])
    if isinstance(coff_list[0], list):
        matrix_list = []
        for v in coff_list[0]:
            matrix_list.append(_create_uniform_mat(v, embed_size))
        return matrix_list
    else:
        return _create_uniform_mat(coff_list[0], embed_size)


def _create_seq_weight(value):
    max_item_size = _get_max_item_size(value.name)
    seq_weight = np.zeros([max_item_size, max_item_size])
    for i in range(max_item_size):
        for j in range(i + 1):
            seq_weight[i][j] = 1 / (i + 1.0)
    return tf.constant(seq_weight, dtype=tf.float32)


def embed_seq_weight(coff_list):
    if isinstance(coff_list[0], list):
        matrix_list = [_create_seq_weight(v) for v in coff_list[0]]
        return matrix_list
    else:
        return _create_seq_weight(coff_list[0])


def ffm_embed_matrix(coff_list):
    embed_size = int(coff_list[1])
    stddev = coff_list[2] if len(coff_list) > 2 else 1.0 / math.sqrt(embed_size)
    value_list = coff_list[0]
    matrix_list = []
    for i in range(0, len(value_list) - 1):
        for j in range(i + 1, len(value_list)):
            matrix_list.append(_create_mat(value_list[i], embed_size, stddev))
            matrix_list.append(_create_mat(value_list[j], embed_size, stddev))
    return matrix_list


def l2_normalize(coff_list):
    v = coff_list[0]
    dim = int(coff_list[1])
    if isinstance(v, list):
        r = tf.nn.l2_loss(v[0])
        for i in range(1, len(v)):
            r = r + tf.nn.l2_normalize(v[i], axis=dim)
        return r
    else:
        return tf.nn.l2_normalize(v, axis=dim)


def l2_loss(coff_list):
    v = coff_list[0]
    if isinstance(v, list):
        r = tf.nn.l2_loss(v[0])
        for i in range(1, len(v)):
            r = r + tf.nn.l2_loss(v[i])
        return r
    else:
        return tf.nn.l2_loss(v)


def l1_loss(coff_list):
    v = coff_list[0]
    if isinstance(v, list):
        r = tf.reduce_mean(tf.abs(tf.reshape(v[0], [-1])), 0)
        for i in range(1, len(v)):
            r = r + tf.reduce_mean(tf.abs(tf.reshape(v[i], [-1])), 0)
        return r
    else:
        return tf.reduce_mean(tf.abs(tf.reshape(v, [-1])), 0)


def sampled_softmax_loss(coff_list):
    weights = coff_list[0]
    biases = coff_list[1]
    inputs = coff_list[2]
    labels = coff_list[3]
    num_sampled = int(coff_list[4])
    num_classes = _get_max_item_size(labels.name)
    return tf.nn.sampled_softmax_loss(weights=weights, biases=biases, inputs=inputs,
                                      labels=tf.reshape(labels, [-1, 1]), num_sampled=num_sampled,
                                      num_classes=num_classes)


def nce_loss(coff_list):
    weights = coff_list[0]
    biases = coff_list[1]
    inputs = coff_list[2]
    labels = coff_list[3]
    num_sampled = int(coff_list[4])  # 负样本数
    num_classes = _get_max_item_size(labels.name)
    return tf.nn.nce_loss(weights=weights, biases=biases, inputs=inputs, labels=tf.reshape(labels, [-1, 1]),
                          num_sampled=num_sampled, num_classes=num_classes)


def add_n(coff_list):
    c_list = [v for v in coff_list[0]]
    return tf.add_n(c_list)


def dot(coff_list):
    v0 = coff_list[0]
    v1 = coff_list[1]
    if isinstance(v0, list) and isinstance(v1, list):
        return [tf.reshape(tf.reduce_sum(v0[i] * v1[i], axis=1), shape=[-1, 1]) for i in range(len(v0))]
    else:
        return tf.reshape(tf.reduce_sum(v0 * v1, axis=1), shape=[-1, 1])


def concat(coff_list):
    tf_list = [c for c in coff_list[0]]
    return tf.concat(tf_list, axis=int(coff_list[1]))


def matmul(coff_list):
    return tf.matmul(coff_list[0], coff_list[1])


def squeeze(coff_list):
    if isinstance(coff_list[0], list):
        return [tf.squeeze(c) for c in coff_list[0]]
    else:
        return tf.squeeze(coff_list[0])


def expand_dims(coff_list):
    dim = int(coff_list[1])
    if isinstance(coff_list[0], list):
        return [tf.expand_dims(c, dim) for c in coff_list[0]]
    else:
        return tf.expand_dims(coff_list[0], dim)


def transpose(coff_list):
    if isinstance(coff_list[0], list):
        return [tf.transpose(c) for c in coff_list[0]]
    else:
        return tf.transpose(coff_list[0])


def clip_by_value(coff_list):
    sv = coff_list[1]
    ev = coff_list[2]
    if isinstance(coff_list[0], list):
        return [tf.clip_by_value(c, sv, ev) for c in coff_list[0]]
    else:
        return tf.clip_by_value(coff_list[0], sv, ev)


def sigmoid(coff_list):
    return tf.nn.sigmoid(coff_list[0])


def softmax(coff_list):
    return tf.nn.softmax(coff_list[0])


def relu(coff_list):
    return tf.nn.relu(coff_list[0])


def tanh(coff_list):
    return tf.nn.tanh(coff_list[0])


def kernel_fm(coff_list):
    tf_list = [c for c in coff_list[0]]
    kernel_mat = coff_list[1]
    res_list = []
    pid = 0
    for i in range(0, len(tf_list) - 1):
        for j in range(i + 1, len(tf_list)):
            k_mat = tf.gather(kernel_mat, pid)
            res_list.append(tf.reshape(tf.reduce_sum(tf.matmul(tf_list[i], k_mat) * tf_list[j], axis=1), shape=[-1, 1]))
    return res_list


def fm(coff_list):
    tf_list = [c for c in coff_list[0]]
    res_list = []
    for i in range(0, len(tf_list) - 1):
        for j in range(i + 1, len(tf_list)):
            res_list.append(tf.reshape(tf.reduce_sum(tf_list[i] * tf_list[j], axis=1), shape=[-1, 1]))
    return res_list


def ffm(coff_list):
    tf_list = [c for c in coff_list[0]]
    res_list = []
    for i in range(0, len(tf_list), 2):
        res_list.append(tf.reshape(tf.reduce_sum(tf_list[i] * tf_list[i + 1], axis=1), shape=[-1, 1]))
    return res_list


def sigmoid_cross_entropy_with_logits(coff_list):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=coff_list[0], logits=coff_list[1])


def cross_entropy_with_logits(coff_list):
    # 注意没有sigmoid
    z = coff_list[0]
    x = coff_list[1]
    return z * -tf.log(x) + (1 - z) * -tf.log(1 - x)


def create_char_opt():
    opt_dict = {
        "len": length,
        "adam_optim": adam_optim,
        "grad_optim": grad_optim,
        "ftrl_optim": ftrl_optim,
        "minimize": minimize,
        "log": log,
        "l2_normalize": l2_normalize,
        "l2_loss": l2_loss,
        "l1_loss": l1_loss,
        "nce_loss": nce_loss,
        "sampled_softmax_loss": sampled_softmax_loss,
        "add_n": add_n,
        "dot": dot,
        "dropout": dropout,
        "matrix": matrix,
        "array": array,
        "get": get_value,
        "set": set_value,
        "reduce_sum": reduce_sum,
        "reduce_mean": reduce_mean,
        "embedding_lookup": embedding_lookup,
        "ffm_embedding_lookup": ffm_embedding_lookup,
        "reshape": reshape,
        "concat": concat,
        "matmul": matmul,
        "transpose": transpose,
        "squeeze": squeeze,
        "expand_dims": expand_dims,
        "clip_by_value": clip_by_value,
        "sigmoid": sigmoid,
        "softmax": softmax,
        "relu": relu,
        "tanh": tanh,
        "fm": fm,
        "kernel_fm": kernel_fm,
        "ffm": ffm,
        "one_hot": one_hot,
        "variable": variable,
        "embed_matrix": embed_matrix,
        "embed_uniform_matrix": embed_uniform_matrix,
        "embed_seq_weight": embed_seq_weight,
        "ffm_embed_matrix": ffm_embed_matrix,
        "sigmoid_cross_entropy_with_logits": sigmoid_cross_entropy_with_logits,
        "cross_entropy_with_logits": cross_entropy_with_logits,
    }
    return opt_dict
