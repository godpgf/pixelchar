from .char_model import CharModel
import numpy as np


class CharClassificationModel(CharModel):
    def __init__(self, data_meta_dict, model_text):
        super(CharClassificationModel, self).__init__(data_meta_dict, model_text)

    def fit(self, train_data_source_factory, eval_data_source_factory=None, train_loss_name="loss",
            eval_loss_name="loss", label_name="label", p_label_name="predict", epoch_num="epoch_num", optim_name="train_optimzer",
            char_eval_list=None):
        res = None
        epoch_num = int(self.db[epoch_num])
        fit_name_list = [train_loss_name, label_name]
        if p_label_name is not None:
            fit_name_list.append(p_label_name)
        train_data_name_list = self._get_data_name_list(fit_name_list)
        train_data_source = train_data_source_factory(train_data_name_list)
        if eval_data_source_factory is not None and p_label_name is not None:
            eval_data_name_list = self._get_data_name_list(fit_name_list)
            eval_data_source = eval_data_source_factory(eval_data_name_list)
        with self.sess.as_default():
            with self.graph.as_default():
                self.initialize()
                for epoch_index in range(epoch_num):
                    train_data_iter = iter(train_data_source)
                    while True:
                        try:
                            feed_dict = self.feed_data(train_data_iter, train_data_name_list)
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
                                    char_eval.push(loss, predict, label)
                        except StopIteration as e:
                            break

                    if eval_data_source_factory is not None and p_label_name is not None:
                        eval_data_iter = iter(eval_data_source)
                        while True:
                            try:
                                feed_dict = self.feed_data(eval_data_iter, eval_data_name_list)
                                label = feed_dict[self.db[label_name]]
                                loss, predict = self.sess.run([self.db[eval_loss_name], self.db[p_label_name]],
                                                           feed_dict=feed_dict)
                                if char_eval_list is not None:
                                    for char_eval in char_eval_list:
                                        char_eval.push(loss, predict, label)
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
                        p_list.append(self.sess.run(self.db[p_label_name], feed_dict=self.feed_data(data_iter, data_name_list)))
                    except StopIteration as e:
                        break
                return np.concatenate(p_list, axis=0)
