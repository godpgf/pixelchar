from rpn import rpn_encode
import tensorflow as tf
from .char_opt import create_char_opt
from pixelchar.data import *
import os


def check_path(ckpt_file_path):
    path = os.path.dirname(os.path.abspath(ckpt_file_path))
    if os.path.isdir(path) is False:
        os.makedirs(path)


class CharModel(object):
    def __init__(self, data_meta_dict, model_text):
        self.data_meta_dict = data_meta_dict
        self.graph = tf.Graph()
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=sess_config)
        self.db = dict()
        self.opt_dict = create_char_opt()
        self.rpn_opt_dict = {}
        self.is_load = False
        with self.sess.as_default():
            with self.graph.as_default():
                self._create_placeholder()
                self._create_model(model_text)

    def clear(self):
        self.is_load = False

    def _create_placeholder(self):
        for key, value in self.data_meta_dict.items():
            if isinstance(value, ValueDataMeta):
                self.db[key] = tf.placeholder(dtype=tf.float32 if value.value_type == 'float32' else tf.int32,
                                              shape=[None], name=key)
            else:
                assert isinstance(value, SeqDataMeta)
                name = "%s_%d" % (key, value.max_item_size)
                if value.seq_length > 0:
                    self.db[key] = tf.placeholder(dtype=tf.int32, shape=[None, value.seq_length], name=name)
                else:
                    self.db[key] = tf.placeholder(dtype=tf.int32, shape=[None], name=name)
        # 关键字
        self.db["epoch_idx"] = tf.placeholder(dtype=tf.int32, shape=None, name="epoch_idx")

    def _create_model(self, model_text):
        # 通过文本来初始化模型结构
        model_text = model_text.replace("\r", "").replace("\t", "").replace(" ", "")
        line_list = model_text.split('\n')
        for line in line_list:
            if len(line) > 1 and not line.startswith("#"):
                self._create_sub_model(line)

    def _create_sub_model(self, line):
        rpn_opt_list = rpn_encode(line)
        self.rpn_opt_dict[rpn_opt_list[0]] = rpn_opt_list
        coff_list = []
        for opt in rpn_opt_list:
            if opt == '+' or opt == '-' or opt == '*' or opt == '/' or opt == '=':
                # 如果遇到操作符
                v2 = coff_list.pop()
                v1 = coff_list.pop()
                if opt == '+':
                    res = v1 + v2
                elif opt == '-':
                    res = v1 - v2
                elif opt == '*':
                    res = v1 * v2
                elif opt == '/':
                    res = v1 / v2
                else:
                    assert opt == '='
                    if isinstance(v1, tuple):
                        # 设置数组中的某个元素
                        value_array = v1[0]
                        value_index = int(v1[1])
                        value_array[value_index] = v2
                        res = v2
                    else:
                        self.db[v1] = v2
                        res = v2
                coff_list.append(res)
            elif opt == "@":
                # 遇到函数开始标记
                coff_list.append("@")
            elif not isinstance(opt, float) and opt.startswith("@"):
                # 如果遇到函数
                fun_coff_list = []
                while coff_list[-1] != "@":
                    fun_coff_list.append(coff_list.pop())
                fun_coff_list.reverse()
                coff_list.pop()
                opt = opt[1:]
                res = self.opt_dict[opt](fun_coff_list)
                coff_list.append(res)
            else:
                # 剩下的只有操作数
                if isinstance(opt, float):
                    coff_list.append(opt)
                else:
                    coff_list.append(self.db.get(opt, opt))

    # 推理模型所需要的数据源名字列表
    def _get_data_name_list(self, res_name_list):
        # 这些操作用到的数据不作为数据源（因为仅仅使用了meta信息）
        exclude_opt = set(["@embed_matrix", "@embed_seq_weight", "@embed_neg_bias", "@ffm_embed_matrix", "@variable", "@embed_uniform_matrix"])
        name_set = set()
        finish_set = set()
        while len(res_name_list):
            res_name = res_name_list.pop()
            finish_set.add(res_name)
            if res_name in self.data_meta_dict:
                name_set.add(res_name)
            elif res_name in self.rpn_opt_dict:
                for i in range(len(self.rpn_opt_dict[res_name])-1, 0, -1):
                    opt = self.rpn_opt_dict[res_name][i]
                    if opt in exclude_opt:
                        break
                    if isinstance(opt, float) or opt == '+' or opt == '-' or opt == '*' or opt == '/' or opt == '=' or opt == '@' or opt.startswith(
                            '@'):
                        continue
                    if opt in self.data_meta_dict:
                        name_set.add(opt)
                    elif opt in self.rpn_opt_dict and opt not in finish_set:
                        res_name_list.append(opt)
        return list(name_set)

    def feed_data(self, data_iter, data_name_list, attach_data_size=0):
        data_value_list = next(data_iter)

        # 切分出附加数据和需要填写到模型里面的数据-------------------------------------------------
        attach_data_list = data_value_list[-attach_data_size:] if attach_data_size > 0 else None
        data_value_list = data_value_list[:-attach_data_size] if attach_data_size > 0 else data_value_list
        data_name_list = data_name_list[:-attach_data_size] if attach_data_size > 0 else data_name_list

        feed_dict = {self.db[data_name]: data_value for data_value, data_name in zip(data_value_list, data_name_list)}
        return feed_dict, attach_data_list

    def initialize(self):
        if not self.is_load:
            self.sess.run(tf.global_variables_initializer())
            self.is_load = True

    def save(self, ckpt_file_path):
        check_path(ckpt_file_path)
        with self.sess.as_default():
            with self.graph.as_default():
                tf.train.Saver().save(self.sess, ckpt_file_path)

    def load(self, ckpt_file_path):
        with self.sess.as_default():
            with self.graph.as_default():
                tf.train.Saver().restore(self.sess, ckpt_file_path)
                self.is_load = True
