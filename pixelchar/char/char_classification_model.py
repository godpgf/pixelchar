from .char_model import CharModel
import numpy as np
import tensorflow as tf


class CharClassificationModel(CharModel):
    def __init__(self, data_meta_dict, model_text):
        super(CharClassificationModel, self).__init__(data_meta_dict, model_text)

    def _create_placeholder(self):
        super(CharClassificationModel, self)._create_placeholder()
        # 关键字
        self.db["epoch_idx"] = tf.placeholder(dtype=tf.int32, shape=None, name="epoch_idx")

    def fit(self, train_data_source_factory, eval_data_source_factory=None, train_loss_name="loss",
            eval_loss_name="loss", label_name="label", p_label_name="predict", epoch_num="epoch_num",
            optim_name="train_optimzer", char_eval_list=None, attach_data_name_list=None):
        res = None
        epoch_num = int(self.db[epoch_num])

        # 推导出为了计算train_loss_name, label_name，p_label_name所需要的数据名字
        fit_name_list = [train_loss_name, label_name]
        if p_label_name is not None:
            fit_name_list.append(p_label_name)
        train_data_name_list = self._get_data_name_list(fit_name_list)

        # 除了计算需要的数据，还需要一些附加数据，用来计算一些特殊的评估指标
        attach_data_size = 0 if attach_data_name_list is None else len(attach_data_name_list)
        if attach_data_size > 0:
            train_data_name_list.extend(attach_data_name_list)

        train_data_source = train_data_source_factory(train_data_name_list)

        # 同理，得到验证需要的数据
        if eval_data_source_factory is not None:
            fit_name_list = [label_name]
            if p_label_name is not None:
                fit_name_list.append(p_label_name)
            else:
                fit_name_list.append(train_loss_name)
            eval_data_name_list = self._get_data_name_list(fit_name_list)
            if attach_data_size > 0:
                eval_data_name_list.extend(attach_data_name_list)
            eval_data_source = eval_data_source_factory(eval_data_name_list)

        with self.sess.as_default():
            with self.graph.as_default():
                self.initialize()
                for epoch_index in range(epoch_num):
                    train_data_iter = iter(train_data_source)
                    while True:
                        try:
                            feed_dict, attach_data_list = self.feed_data(train_data_iter, train_data_name_list,
                                                                         attach_data_size)
                            feed_dict[self.db["epoch_idx"]] = [epoch_index]
                            label = feed_dict[self.db[label_name]]
                            if p_label_name is not None:
                                loss, _, predict = self.sess.run(
                                    [self.db[train_loss_name], self.db[optim_name], self.db[p_label_name]],
                                    feed_dict=feed_dict)
                            else:
                                loss, _ = self.sess.run(
                                    [self.db[train_loss_name], self.db[optim_name]],
                                    feed_dict=feed_dict)
                                predict = None
                            if eval_data_source_factory is None and char_eval_list is not None:
                                for char_eval in char_eval_list:
                                    char_eval.push(loss, predict, label, attach_data_list)
                        except StopIteration as e:
                            break

                    if eval_data_source_factory is not None:
                        eval_data_iter = iter(eval_data_source)
                        while True:
                            try:
                                feed_dict, attach_data_list = self.feed_data(eval_data_iter, eval_data_name_list,
                                                                             attach_data_size)
                                feed_dict[self.db["epoch_idx"]] = [epoch_index]
                                label = feed_dict[self.db[label_name]]
                                if p_label_name is not None:
                                    loss, predict = self.sess.run([self.db[eval_loss_name], self.db[p_label_name]],
                                                                  feed_dict=feed_dict)
                                else:
                                    loss = self.sess.run(
                                        self.db[train_loss_name],
                                        feed_dict=feed_dict)[0]
                                    predict = None
                                if char_eval_list is not None:
                                    for char_eval in char_eval_list:
                                        char_eval.push(loss, predict, label, attach_data_list)
                            except StopIteration as e:
                                break

                    if char_eval_list is not None:
                        res_list = [char_eval.pop() for char_eval in char_eval_list]
                        res = "epoch %d\t%s" % (epoch_index, "\t".join(res_list))
                        print(res)
        return res

    def predict(self, data_source_factory, p_label_name="predict"):
        data_name_list = self._get_data_name_list([p_label_name])
        data_source = data_source_factory(data_name_list)
        data_iter = iter(data_source)
        with self.sess.as_default():
            with self.graph.as_default():
                p_list = []
                while True:
                    try:
                        feed_dict, _ = self.feed_data(data_iter, data_name_list)
                        p_list.append(self.sess.run(self.db[p_label_name], feed_dict=feed_dict))
                    except StopIteration as e:
                        break
                return np.concatenate(p_list, axis=0)

    def get_value(self, value_name):
        with self.sess.as_default():
            with self.graph.as_default():
                return self.sess.run(self.db[value_name])
