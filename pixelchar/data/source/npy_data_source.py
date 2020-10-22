from .data_source import *
import numpy as np
import os


class NPYTrainDataSource(TrainDataSource):
    def __init__(self, npy_path, weight_name, data_name_list, max_batch_size=None, index_size=100000000):
        self.npy_path = npy_path
        super(NPYTrainDataSource, self).__init__(weight_name, data_name_list, max_batch_size, index_size)

    def _get_weight(self, weight_name):
        return np.load(os.path.join(self.npy_path, weight_name + ".npy"))

    def _get_data_list(self):
        return [np.load(os.path.join(self.npy_path, data_name + ".npy")) for data_name in self.data_name_list]


class NPYEvalDataSource(EvalDataSource):
    def __init__(self, npy_path, data_name_list, max_batch_size=None):
        self.npy_path = npy_path
        super(NPYEvalDataSource, self).__init__(data_name_list, max_batch_size)

    def _get_data_list(self):
        return [np.load(os.path.join(self.npy_path, data_name + ".npy")) for data_name in self.data_name_list]


class NPYNegItemTrainDataSource(NegItemTrainDataSource):
    def __init__(self, npy_path, weight_name, item_name, label_name, data_name_list, neg_power=0.25,
                 negative_sample_scale=20, max_batch_size=None):
        self.npy_path = npy_path
        super(NPYNegItemTrainDataSource, self).__init__(weight_name, item_name, label_name, data_name_list, neg_power,
                                                        negative_sample_scale, max_batch_size)

    def _get_weight(self, weight_name):
        return np.load(os.path.join(self.npy_path, weight_name + ".npy"))

    def _get_data_list(self, label_name):
        return [np.load(os.path.join(self.npy_path, data_name + ".npy")) if data_name != label_name else None for
                data_name in self.data_name_list]
