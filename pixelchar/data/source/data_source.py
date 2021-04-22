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
                    res = [self.data_sour.data_dict[data_name][self.read_data_num:self.batch_size] for data_name in
                           self.data_sour.data_name_list]
                    self.read_data_num += self.batch_size
                else:
                    res = [self.data_sour.data_dict[data_name][self.read_data_num] for data_name in
                           self.data_sour.data_name_list]
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


class TrainDataSource(DataSource):
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

    def __init__(self, weight_name, data_name_list, max_batch_size=None, index_size=100000000):
        super(TrainDataSource, self).__init__(data_name_list, max_batch_size)
        self.data_list = self._get_data_list()
        self.index = self.weight_2_index(self._get_weight(weight_name), index_size)

    def _get_weight(self, weight_name):
        pass

    def _get_data_list(self):
        return self

    def __len__(self):
        return len(self.data_list[0])

    def __iter__(self):
        return self.DataIterator(self.index, self.data_list, self.max_batch_size)


class DictTrainDataSource(TrainDataSource):
    def __init__(self, data_package, weight_name, data_name_list, max_batch_size=None, index_size=100000000):
        self.data_package = data_package
        super(DictTrainDataSource, self).__init__(weight_name, data_name_list, max_batch_size, index_size)

    def _get_weight(self, weight_name):
        return self.data_package[weight_name]

    def _get_data_list(self):
        return [self.data_package[data_name] for data_name in self.data_name_list]


class PairDictTrainDataSource(DataSource):
    class PairDataIterator(object):
        def __init__(self, pair_data_source):
            max_batch_size = pair_data_source.max_batch_size
            self.read_data_num = 0
            self.pair_data_source = pair_data_source
            self.index = np.empty(max_batch_size, np.int32)
            self.pair_index = np.empty(max_batch_size, np.int32)
            self.label = np.empty(max_batch_size, np.float32)
            self.batch_size = max_batch_size

        def __next__(self):
            if self.read_data_num >= len(self.pair_data_source):
                raise StopIteration
            else:
                # 随机选择flag
                flag_batch = np.random.choice(self.pair_data_source.flag, self.batch_size, replace=True)
                # 设置index
                for i, flag in enumerate(flag_batch):
                    rand_ids = np.random.choice(self.pair_data_source.flag2rand_ids[flag], 2, replace=True)
                    ids = self.pair_data_source.flag2ids[flag][rand_ids]
                    self.index[i] = ids[0]
                    self.pair_index[i] = ids[1]
                    self.label[i] = (1.0 if self.pair_data_source.rank_array[ids[0]] > self.pair_data_source.rank_array[
                        ids[1]] else 0.0)

                self.read_data_num += self.batch_size
                return [self.label if data is None else (data[self.pair_index] if is_pair else data[self.index]) for
                        data, is_pair in zip(self.pair_data_source.data_list, self.pair_data_source.is_pair_list)]

    def __init__(self, data_package, weight_name, label_name, flag_name, rank_value_name, data_name_list,
                 max_batch_size):
        self.data_package = data_package
        super(PairDictTrainDataSource, self).__init__(data_name_list, max_batch_size=max_batch_size)
        weight_array = self._get_weight(weight_name)
        flag_array = data_package[flag_name]
        self.flag = np.array(list(set(flag_array.tolist())))
        self.flag2ids = {flag: np.where(flag_array == flag)[0] for flag in self.flag}
        self.flag2rand_ids = {flag: self.weight_2_index(weight_array[ids], len(ids) * 16) for flag, ids in
                              self.flag2ids.items()}
        self.data_list = self._get_data_list(label_name)
        self.is_pair_list = [data_name.startswith("pair_") for data_name in self.data_name_list]
        self.rank_array = data_package[rank_value_name]

    def _get_weight(self, weight_name):
        return self.data_package[weight_name]

    def _get_data_list(self, label_name):
        return [self.data_package[data_name.replace("pair_", "")] if data_name != label_name else None for data_name in
                self.data_name_list]

    def __len__(self):
        return len(self.rank_array)

    def __iter__(self):
        return self.PairDataIterator(self)


class EvalDataSource(DataSource):
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

    def __init__(self, data_name_list, max_batch_size=None):
        super(EvalDataSource, self).__init__(data_name_list, max_batch_size)
        self.data_list = self._get_data_list()

    def _get_data_list(self):
        return self

    def __len__(self):
        return len(self.data_list[0])

    def __iter__(self):
        return self.DataIterator(self.data_list, self.max_batch_size)


class DictEvalDataSource(EvalDataSource):
    def __init__(self, data_package, data_name_list, max_batch_size=None):
        self.data_package = data_package
        super(DictEvalDataSource, self).__init__(data_name_list, max_batch_size)

    def _get_data_list(self):
        return [self.data_package[data_name] for data_name in self.data_name_list]


class NegItemTrainDataSource(DataSource):
    class DataIterator(object):
        def __init__(self, index, neg_index, negative_sample_scale, data_list, data_cache_list,
                     item_data_index, max_batch_size):
            self.read_data_num = 0
            self.index = index
            self.neg_index = neg_index
            self.negative_sample_scale = negative_sample_scale
            self.data_list = data_list
            self.data_cache_list = data_cache_list
            self.batch_size = max_batch_size
            self.item_data_index = item_data_index
            for i in range(len(self.data_list)):
                if self.data_list[i] is not None:
                    self.valid_id = i
                    break

        def __next__(self):
            if self.read_data_num >= len(self.data_list[self.valid_id]):
                raise StopIteration
            else:
                cur_index = np.random.choice(self.index, self.batch_size, replace=True)
                cur_neg_index = np.random.choice(self.neg_index, self.batch_size * self.negative_sample_scale,
                                                 replace=True)
                self.read_data_num += self.batch_size

                for id, data in enumerate(self.data_list):
                    if id in self.item_data_index:
                        self.data_cache_list[id][:self.batch_size] = data[cur_index]
                        self.data_cache_list[id][self.batch_size:] = data[cur_neg_index]
                    elif data is None:
                        pass
                    else:
                        p_data = data[cur_index]
                        for i in range(self.negative_sample_scale + 1):
                            self.data_cache_list[id][i * self.batch_size:(i + 1) * self.batch_size] = p_data
                return self.data_cache_list

    def __init__(self, weight_name, label_name, data_name_list, neg_data_name_list, neg_power=0.25,
                 negative_sample_scale=20, max_batch_size=None):
        super(NegItemTrainDataSource, self).__init__(data_name_list, max_batch_size)
        weight = self._get_weight(weight_name)
        self.index = self.weight_2_index(weight)
        self.data_list = self._get_data_list(label_name)

        label = np.zeros((negative_sample_scale + 1) * max_batch_size)
        label[:max_batch_size] = 1.0
        self.data_cache_list = []
        for data in self.data_list:
            if data is None:
                self.data_cache_list.append(label)
            else:
                shape = list(data.shape)
                shape[0] = len(label)
                self.data_cache_list.append(np.empty(shape, data.dtype))

        # 得到负样本
        self.item_data_index = []
        neg_data_name_set = set(neg_data_name_list)
        for id, data_name in enumerate(data_name_list):
            if data_name in neg_data_name_set:
                self.item_data_index.append(id)
        self.item_data_index = set(self.item_data_index)

        assert len(self.item_data_index) > 0
        self.neg_index = self.weight_2_index(np.power(weight, neg_power))
        self.negative_sample_scale = negative_sample_scale

    def __len__(self):
        return len(self.data_list[self.item_data_index])

    def __iter__(self):
        return self.DataIterator(self.index, self.neg_index, self.negative_sample_scale, self.data_list,
                                 self.data_cache_list, self.item_data_index, self.max_batch_size)

    def _get_weight(self, weight_name):
        return self

    def _get_data_list(self, label_name):
        return self


class DictNegItemTrainDataSource(NegItemTrainDataSource):
    def __init__(self, data_package, weight_name, label_name, data_name_list, neg_data_name_list, neg_power=0.25,
                 negative_sample_scale=20, max_batch_size=None):
        self.data_pacakge = data_package
        super(DictNegItemTrainDataSource, self).__init__(weight_name, label_name, data_name_list, neg_data_name_list,
                                                         neg_power, negative_sample_scale, max_batch_size)

    def _get_weight(self, weight_name):
        return self.data_pacakge[weight_name]

    def _get_data_list(self, label_name):
        return [self.data_pacakge[data_name] if data_name != label_name else None for data_name in self.data_name_list]
