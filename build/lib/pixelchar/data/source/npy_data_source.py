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

    def __init__(self, data_package, weight_name, data_name_list, max_batch_size=None, index_size=100000000):
        super(NPYTrainDataSource, self).__init__(data_name_list, max_batch_size)
        if isinstance(data_package, dict):
            weight = data_package[weight_name]
            self.data_list = [data_package[data_name] for data_name in data_name_list]
        else:
            weight = np.load(data_package + weight_name + ".npy")
            self.data_list = [np.load(data_package + data_name + ".npy") for data_name in data_name_list]
        self.index = self.weight_2_index(weight, index_size)

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

    def __init__(self, data_package, data_name_list, max_batch_size=None):
        super(NPYEvalDataSource, self).__init__(data_name_list, max_batch_size)
        if isinstance(data_package, dict):
            self.data_list = [data_package[data_name] for data_name in data_name_list]
        else:
            self.data_list = [np.load(data_package + data_name + ".npy") for data_name in data_name_list]

    def __len__(self):
        return len(self.data_list[0])

    def __iter__(self):
        return self.DataIterator(self.data_list, self.max_batch_size)


class NPYNegItemTrainDataSource(DataSource):
    class DataIterator(object):
        def __init__(self, index, neg_item, neg_index, negative_sample_scale, data_list, item_data_index, max_batch_size):
            self.read_data_num = 0
            self.index = index
            self.neg_item = neg_item
            self.neg_index = neg_index
            self.negative_sample_scale = negative_sample_scale
            self.data_list = data_list
            self.batch_size = max_batch_size
            self.label = np.zeros((negative_sample_scale + 1) * max_batch_size)
            self.label[:max_batch_size] = 1.0
            self.item_data_index = item_data_index

        def __next__(self):
            if self.read_data_num >= len(self.data_list[0]):
                raise StopIteration
            else:
                cur_index = np.random.choice(self.index, self.batch_size, replace=True)
                cur_neg_index = np.random.choice(self.neg_index, self.batch_size * self.negative_sample_scale, replace=True)
                self.read_data_num += self.batch_size
                res = []
                for id, data in enumerate(self.data_list):
                    if id == self.item_data_index:
                        res.append(np.concatenate((data[cur_index], self.neg_item[cur_neg_index]), axis=0))
                    elif data is None:
                        res.append(self.label)
                    else:
                        res.append(np.concatenate([data[cur_index]] * (self.negative_sample_scale + 1), axis=0))
                return res

    def __init__(self, path, weight_name, item_name, label_name, data_name_list, neg_power=0.25, negative_sample_scale=20, max_batch_size=None):
        super(NPYNegItemTrainDataSource, self).__init__(data_name_list, max_batch_size)
        weight = np.load(path + weight_name + ".npy")
        self.index = self.weight_2_index(weight)
        self.data_list = [np.load(path + data_name + ".npy") if data_name != label_name else None for data_name in data_name_list]

        # 得到负样本
        self.item_data_index = None
        for id, data_name in enumerate(data_name_list):
            if data_name == item_name:
                self.item_data_index = id
                break
        assert self.item_data_index is not None
        item_2_weight = {}
        for i, w in zip(self.data_list[self.item_data_index], weight):
            item_2_weight[i] = item_2_weight.get(i, 0.0) + w
        neg_item = []
        neg_weight = []
        for i, w in item_2_weight.items():
            neg_item.append(i)
            neg_weight.append(w)

        self.neg_item = np.array(neg_item)
        self.neg_index = self.weight_2_index(np.power(np.array(neg_weight), neg_power))
        self.negative_sample_scale = negative_sample_scale

    def __len__(self):
        return len(self.data_list[self.item_data_index])

    def __iter__(self):
        return self.DataIterator(self.index, self.neg_item, self.neg_index, self.negative_sample_scale, self.data_list, self.item_data_index, self.max_batch_size)
