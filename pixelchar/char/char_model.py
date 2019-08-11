from rpn import rpn_encode
from .char_opt import *
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
        self.opt_dict = self._create_char_opt()
        self.rpn_opt_dict = {}
        self.is_load = False
        with self.sess.as_default():
            with self.graph.as_default():
                self._create_placeholder()
                self._create_model(model_text)

    def _create_char_opt(self):
        opt_dict = {
            "adam_optim": adam_optim,
            "minimize": minimize,
            "l2_loss": l2_loss,
            "dot": dot,
            "matrix": matrix,
            "array": array,
            "get": get_value,
            "set": set_value,
            "reduce_sum": reduce_sum,
            "reduce_mean": reduce_mean,
            "embedding_lookup": embedding_lookup,
            "reshape": reshape,
            "concat": concat,
            "matmul": matmul,
            "sigmoid": sigmoid,
            "embed_matrix": lambda coff_list: embed_matrix(self.data_meta_dict, coff_list),
            "sigmoid_cross_entropy_with_logits": sigmoid_cross_entropy_with_logits,
        }
        return opt_dict

    def _create_placeholder(self):
        for key, value in self.data_meta_dict.items():
            if isinstance(value, ValueDataMeta):
                self.db[key] = tf.placeholder(dtype=tf.float32 if value.value_type == 'float32' else tf.int32,
                                              shape=[None], name=key)
            elif isinstance(value, BarDataMeta):
                self.db[key] = tf.placeholder(dtype=tf.int32, shape=[None, value.width], name=key)
            elif isinstance(value, SeqDataMeta):
                self.db[key] = tf.placeholder(dtype=tf.int32, shape=[None, value.seq_length], name=key)

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
                    res = v1[1] + v2[1]
                elif opt == '-':
                    res = v1[1] - v2[1]
                elif opt == '*':
                    res = v1[1] * v2[1]
                elif opt == '/':
                    res = v1[1] / v2[1]
                else:
                    res = v2[1]
                    if v1[0] == "set":
                        # 设置数组中的某个元素
                        value_array = v1[1][0]
                        value_index = int(v1[1][1])
                        value_array[value_index] = res
                    else:
                        self.db[v1[0]] = res
                coff_list.append((opt, res))
            elif opt == "@":
                # 遇到函数开始标记
                coff_list.append(("@", None))
            elif not isinstance(opt, float) and opt.startswith("@"):
                # 如果遇到函数
                fun_coff_list = []
                while coff_list[-1][0] != "@":
                    fun_coff_list.append(coff_list.pop())
                fun_coff_list.reverse()
                coff_list.pop()
                opt = opt[1:]
                res = self.opt_dict[opt](fun_coff_list)
                coff_list.append((opt, res))
            else:
                # 剩下的只有操作数
                if isinstance(opt, float):
                    coff_list.append((str(opt), opt))
                else:
                    coff_list.append((opt, self.db.get(opt, None)))

    def _get_data_name_list(self, res_name_list):
        name_set = set()
        finish_set = set()
        while len(res_name_list):
            res_name = res_name_list.pop()
            finish_set.add(res_name)
            if res_name in self.rpn_opt_dict:
                for i in range(1, len(self.rpn_opt_dict[res_name])):
                    opt = self.rpn_opt_dict[res_name][i]
                    if isinstance(opt, float) or opt == '+' or opt == '-' or opt == '*' or opt == '/' or opt == '=' or opt == '@' or opt.startswith(
                            '@'):
                        continue
                    if opt in self.data_meta_dict:
                        name_set.add(opt)
                    elif opt in self.rpn_opt_dict and opt not in finish_set:
                        res_name_list.append(opt)
        return list(name_set)

    def feed_data(self, data_iter, data_name_list):
        data_value_list = next(data_iter)
        feed_dict = {self.db[data_name]: data_value for data_value, data_name in zip(data_value_list, data_name_list)}
        return feed_dict

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
