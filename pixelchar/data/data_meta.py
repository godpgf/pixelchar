import numpy as np
import pickle


class DataMeta(object):
    @classmethod
    def create_bar_data_meta(cls, name, width, height, bar_range_list=None):
        return BarDataMeta(name, width, height, bar_range_list)

    @classmethod
    def create_value_data_meta(cls, name, value_type='float32'):
        return ValueDataMeta(name, value_type)

    @classmethod
    def create_seq_data_meta(cls, name, seq_length, max_item_size):
        return SeqDataMeta(name, seq_length, max_item_size)

    @classmethod
    def create_std_bar_data_meta(cls, name, width, height, std):
        bar_range_list = [-3 * std, -2.5 * std, -2*std]
        sub_bar_num = height - 6
        for i in range(1, sub_bar_num):
            bar_range_list.append(4 * std * i / sub_bar_num - 2 * std)
        bar_range_list.extend([2 * std, 2.5 * std, 3 * std])
        return cls.create_bar_data_meta(name, width, height, bar_range_list)

    @classmethod
    def create_continuous_bar_data_meta(cls, name, height, data, is_avg_dis=True):
        width = data.shape[1]
        data = np.reshape(np.array([(np.max(d), np.min(d)) for d in data]), newshape=[-1])
        argsort_ids = np.argsort(data)
        bar_range_list = []
        if height < 6 or is_avg_dis:
            for i in range(1, height):
                bar_range_list.append(data[argsort_ids[(i * len(data)) // height]])
        else:
            for i in range(1, 3):
                bar_range_list.append(data[argsort_ids[(i * len(data)) // height]])

            sub_height = height - 4
            start_value = data[argsort_ids[2 * len(data) // height]]
            end_value = data[argsort_ids[((height-2) * len(data)) // height]]
            for i in range(1, sub_height):
                bar_range_list.append(start_value + i / sub_height * (end_value - start_value))

            for i in range(height-2, height):
                bar_range_list.append(data[argsort_ids[(i * len(data)) // height]])

        return BarDataMeta(name, width, height, bar_range_list)

    @classmethod
    def save(cls, data_meta_dict, path):
        sd = dict()
        for key, value in data_meta_dict.items():
            if isinstance(value, BarDataMeta):
                sd["%s@BarDataMeta" % key] = (value.name, value.width, value.height, value.bar_range_list)
            elif isinstance(value, ValueDataMeta):
                sd["%s@ValueDataMeta" % key] = (value.name, value.value_type)
            elif isinstance(value, SeqDataMeta):
                sd["%s@SeqDataMeta" % key] = (value.name, value.seq_length, value.max_item_size)
        with open(path, "wb") as f:
            pickle.dump(sd, f)

    @classmethod
    def load(cls, path):
        data_meta_dict = dict()
        with open(path, "rb") as f:
            sd = pickle.load(f)
            for key, value in sd.items():
                tmp = key.split('@')
                if tmp[1] == "BarDataMeta":
                    data_meta_dict[tmp[0]] = cls.create_bar_data_meta(value[0], value[1], value[2], value[3])
                elif tmp[1] == "ValueDataMeta":
                    data_meta_dict[tmp[0]] = cls.create_value_data_meta(value[0], value[1])
                elif tmp[1] == "SeqDataMeta":
                    data_meta_dict[tmp[0]] = cls.create_seq_data_meta(value[0], value[1], value[2])
        return data_meta_dict


class ValueDataMeta(object):
    def __init__(self, name, value_type='float32'):
        self.name = name
        self.value_type = value_type


class SeqDataMeta(object):
    def __init__(self, name, seq_length, max_item_size):
        self.name = name
        self.seq_length = seq_length
        self.max_item_size = max_item_size


# 记录数据源的结构
class BarDataMeta(object):
    def __init__(self, name, width, height, bar_range_list=None):
        self.name = name
        self.width = width
        self.height = height
        self.bar_range_list = bar_range_list
        if bar_range_list is not None:
            assert len(bar_range_list) + 1 == height

    def process_continuous_data(self, data):
        ids = np.empty(data.shape, np.int32)
        # 将连续数据转化成离散的
        def get_p_id(v, offset):
            for i in range(len(self.bar_range_list)):
                if v < self.bar_range_list[i]:
                    return i + offset * self.height
            return self.height - 1 + offset * self.height
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ids[i][j] = get_p_id(data[i][j], j)
        return ids



