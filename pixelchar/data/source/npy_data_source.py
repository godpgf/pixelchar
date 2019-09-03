from .data_source import DataSource
import numpy as np


class NPYTrainDataSource(DataSource):
    class DataIterator(object):
        def __init__(self, index, data_list, max_batch_size):
            self.read_data_num = 0
            self.index = index
            self.data_list = data_list
            self.batch_size = max_batch_size

        def __next__(self):
            if self.read_data_num >= len(self.data_list[0]):
                raise StopIteration
            else:
                cur_index = np.random.choice(self.index, self.batch_size, replace=True)
                self.read_data_num += self.batch_size
                return [data[cur_index] for data in self.data_list]

    def __init__(self, path, weight_name, data_name_list, max_batch_size=None):
        super(NPYTrainDataSource, self).__init__(data_name_list, max_batch_size)
        weight = np.load(path + weight_name + ".npy")
        self.index = self.weight_2_index(weight)
        self.data_list = [np.load(path + data_name + ".npy") for data_name in data_name_list]

    def __len__(self):
        return len(self.data_list[0])

    def __iter__(self):
        return self.DataIterator(self.index, self.data_list, self.max_batch_size)


class NPYEvalDataSource(DataSource):
    class DataIterator(object):
        def __init__(self, data_list, max_batch_size):
            self.read_data_num = 0
            self.data_list = data_list
            self.batch_size = len(data_list[0]) if max_batch_size is None else min(max_batch_size, len(data_list[0]))

        def __next__(self):
            if self.read_data_num >= len(self.data_list[0]):
                raise StopIteration
            else:
                v = [data[self.read_data_num: self.read_data_num + self.batch_size] for data in self.data_list]
                self.read_data_num += self.batch_size
                return v

    def __init__(self, path, data_name_list, max_batch_size=None):
        super(NPYEvalDataSource, self).__init__(data_name_list, max_batch_size)
        self.data_list = [np.load(path + data_name + ".npy") for data_name in data_name_list]

    def __len__(self):
        return len(self.data_list[0])

    def __iter__(self):
        return self.DataIterator(self.data_list, self.max_batch_size)
