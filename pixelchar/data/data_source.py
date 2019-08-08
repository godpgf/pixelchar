import numpy as np


class DataSource(object):
    def __init__(self, data_name_list, max_batch_size=None):
        # 得到所需要的数据名字
        self.data_name_list = data_name_list
        # 最大一次可取出的数据量
        self.max_batch_size = max_batch_size

    @classmethod
    def weight_2_index(cls, weight, index_size=100000000):
        weight /= np.sum(weight)
        for i in range(1, len(weight)):
            weight[i] += weight[i - 1]
        index = np.arange(index_size) / float(index_size)
        cur_weight_index = 0
        for i in range(len(index)):
            if index[i] < weight[cur_weight_index]:
                index[i] = cur_weight_index
            else:
                if cur_weight_index < len(weight) - 1:
                    cur_weight_index += 1
                index[i] = cur_weight_index
        index = index.astype(np.int32)
        return index

    def __iter__(self):
        # 返回数据访问对象
        pass

    def __len__(self):
        # 返回数据总长度
        pass


class TrainDataIterator(object):
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


class NPYTrainDataSource(DataSource):
    def __init__(self, path, weight_name, data_name_list, max_batch_size=None):
        super(NPYTrainDataSource, self).__init__(data_name_list, max_batch_size)
        weight = np.load(path + weight_name + ".npy")
        self.index = self.weight_2_index(weight)
        self.data_list = [np.load(path + data_name + ".npy") for data_name in data_name_list]

    def __len__(self):
        return len(self.data_list[0])

    def __iter__(self):
        return TrainDataIterator(self.index, self.data_list, self.max_batch_size)


class EvalDataIterator(object):
    def __init__(self, data_list, max_batch_size):
        self.read_data_num = 0
        self.data_list = data_list
        self.batch_size = len(data_list[0]) if max_batch_size is None else max_batch_size

    def __next__(self):
        if self.read_data_num >= len(self.data_list[0]):
            raise StopIteration
        else:
            v = [data[self.read_data_num: self.read_data_num + self.batch_size] for data in self.data_list]
            self.read_data_num += self.batch_size
            return v


class NPYEvalDataSource(DataSource):
    def __init__(self, path, data_name_list, max_batch_size=None):
        super(NPYEvalDataSource, self).__init__(data_name_list, max_batch_size)
        self.data_list = [np.load(path + data_name + ".npy") for data_name in data_name_list]

    def __len__(self):
        return len(self.data_list[0])

    def __iter__(self):
        return EvalDataIterator(self.data_list, self.max_batch_size)

