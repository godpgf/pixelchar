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
            while index[i] >= weight[cur_weight_index] and cur_weight_index < len(weight) - 1:
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


class DictDataSource(DataSource):
    class DataIterator(object):
        def __init__(self, data_sour):
            self.data_sour = data_sour
            self.batch_size = data_sour.max_batch_size
            self.read_data_num = 0

        def __next__(self):
            if self.read_data_num >= len(self.data_sour):
                raise StopIteration
            else:
                if self.batch_size is not None:
                    res = [self.data_sour.data_dict[data_name][self.read_data_num:self.batch_size] for data_name in self.data_sour.data_name_list]
                    self.read_data_num += self.batch_size
                else:
                    res = [self.data_sour.data_dict[data_name][self.read_data_num] for data_name in self.data_sour.data_name_list]
                    self.read_data_num += 1
                return res

    def __init__(self, data_dict, data_name_list, max_batch_size=None):
        super(DictDataSource, self).__init__(data_name_list, max_batch_size)
        self.data_dict = data_dict

    def __iter__(self):
        return self.DataIterator(self)

    def __len__(self):
        for key, value in self.data_dict.items():
            return len(value)
        return 0



