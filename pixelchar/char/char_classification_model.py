from .char_model import CharModel
from .char_opt import *


class CharClassificationModel(CharModel):
    def __init__(self, data_meta_dict, model_text):
        super(CharClassificationModel, self).__init__(data_meta_dict, model_text)
        self.is_load = False

    def fit(self, train_data_source_factory, eval_data_source_factory=None, train_loss_name="loss",
            eval_loss_name="loss", label_name="label", p_label_name="predict", epoch_num=8, optim_name="train_optimzer",
            char_eval=None):
        train_data_name_list = self._get_data_name_list([train_loss_name, label_name, p_label_name])
        train_data_source = train_data_source_factory(train_data_name_list)
        if eval_data_source_factory is not None:
            eval_data_name_list = self._get_data_name_list([eval_loss_name, label_name, p_label_name])
            eval_data_source = eval_data_source_factory(eval_data_name_list)
        with self.sess.as_default():
            with self.graph.as_default():
                if not self.is_load:
                    self.sess.run(tf.global_variables_initializer())
                for epoch_index in range(epoch_num):
                    train_data_iter = iter(train_data_source)
                    while True:
                        try:
                            feed_dict = self.feed_data(train_data_iter, train_data_name_list)
                            label = feed_dict[self.db[label_name]]
                            loss, _, predict = self.sess.run(
                                [self.db[train_loss_name], self.db[optim_name], self.db[p_label_name]],
                                feed_dict=feed_dict)
                            if eval_data_source_factory is None and char_eval is not None:
                                char_eval.push(loss, predict, label)
                        except StopIteration as e:
                            break

                    if eval_data_source_factory is not None:
                        eval_data_iter = iter(eval_data_source)
                        while True:
                            try:
                                feed_dict = self.feed_data(eval_data_iter, eval_data_name_list)
                                label = feed_dict[self.db[label_name]]
                                loss, predict = self.sess.run([self.db[eval_loss_name], self.db[p_label_name]],
                                                           feed_dict=feed_dict)
                                if char_eval is not None:
                                    char_eval.push(loss, predict, label)
                            except StopIteration as e:
                                break

                    if char_eval is not None:
                        print("epoch %d\t%s" % (epoch_index, char_eval.pop()))

    def predict(self, data_source_factory, p_label_name="predict"):
        data_name_list = self._get_data_name_list([p_label_name])
        data_source = data_source_factory(data_name_list)
        data_iter = iter(data_source)
        with self.sess.as_default():
            with self.graph.as_default():
                p = self.sess.run(self.db[p_label_name], feed_dict=self.feed_data(data_iter, data_name_list))
                return p
