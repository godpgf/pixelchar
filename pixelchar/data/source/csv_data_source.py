from .data_source import *
from ..meta import *
import numpy as np
import os


class CSVTrainDataSource(TrainDataSource):
    def __init__(self, csv_path, data_meta, weight_name, data_name_list, max_batch_size=None, index_size=100000000):
        self.csv_path = csv_path
        self.data_meta = data_meta
        super(CSVTrainDataSource, self).__init__(weight_name, data_name_list, max_batch_size, index_size)

    def _get_weight(self, weight_name):
        return np.loadtxt(os.path.join(self.csv_path, weight_name + ".csv"), dtype=np.float32)

    def _get_data_list(self):
        data_list = []
        for data_name in self.data_name_list:
            if isinstance(self.data_meta[data_name], ValueDataMeta) and self.data_meta[data_name].value_type == 'float32':
                data_list.append(np.loadtxt(os.path.join(self.csv_path, data_name + ".csv"), delimiter=",", dtype=np.float32))
            else:
                data_list.append(np.loadtxt(os.path.join(self.csv_path, data_name + ".csv"), delimiter=",", dtype=np.int32))
        return data_list


class CSVEvalDataSource(EvalDataSource):
    def __init__(self, csv_path, data_meta, data_name_list, max_batch_size=None):
        self.csv_path = csv_path
        self.data_meta = data_meta
        super(CSVEvalDataSource, self).__init__(data_name_list, max_batch_size)

    def _get_data_list(self):
        data_list = []
        for data_name in self.data_name_list:
            if isinstance(self.data_meta[data_name], ValueDataMeta) and self.data_meta[data_name].value_type == 'float32':
                data_list.append(np.loadtxt(os.path.join(self.csv_path, data_name + ".csv"), delimiter=",", dtype=np.float32))
            else:
                data_list.append(np.loadtxt(os.path.join(self.csv_path, data_name + ".csv"), delimiter=",", dtype=np.int32))
        return data_list


class CSVNegItemTrainDataSource(NegItemTrainDataSource):
    def __init__(self, csv_path, data_meta,  weight_name, label_name, data_name_list, neg_data_name_list, neg_power=0.25,
                 negative_sample_scale=20, max_batch_size=None):
        self.csv_path = csv_path
        self.data_meta = data_meta
        super(CSVNegItemTrainDataSource, self).__init__(weight_name, label_name, data_name_list, neg_data_name_list, neg_power,
                                                        negative_sample_scale, max_batch_size)

    def _get_weight(self, weight_name):
        return np.loadtxt(os.path.join(self.csv_path, weight_name + ".csv"), delimiter=",", dtype=np.float32)

    def _get_data_list(self, label_name):
        data_list = []
        for data_name in self.data_name_list:
            if data_name == label_name:
                data_list.append(None)
            elif isinstance(self.data_meta[data_name], ValueDataMeta) and self.data_meta[data_name].value_type == 'float32':
                data_list.append(np.loadtxt(os.path.join(self.csv_path, data_name + ".csv"), delimiter=",", dtype=np.float32))
            else:
                data_list.append(np.loadtxt(os.path.join(self.csv_path, data_name + ".csv"), delimiter=",", dtype=np.int32))
        return data_list
