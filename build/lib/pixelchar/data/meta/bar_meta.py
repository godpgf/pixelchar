from .base_meta import *
import numpy as np


# 数据是一个连续的序列，比如[[0.1,0.2,0.3,0.1],[0.2,0.1,0.6,0.5]]
# 可以将其按照bar_range_list分桶，比如[[1,3,6,9],[2,4,5,8]]
# 注意，bar[i][t]<bar[i][t+1]
class SeqBarDataMeta(SeqDataMeta):
    def __init__(self, name, width, height, bar_range_list=None):
        super(SeqBarDataMeta, self).__init__(name, width, width * height)
        self.bar_range_list = bar_range_list
        if bar_range_list is not None:
            assert len(bar_range_list) + 1 == height

    def process_continuous_data(self, data):
        # [batch_size, seq_value]-->[batch_size, seq_id]
        ids = np.empty(data.shape, np.int32)
        height = self.max_item_size // self.seq_length
        # width = self.seq_length

        # 将连续数据转化成离散的
        def get_p_id(v, offset):
            for i in range(len(self.bar_range_list)):
                if v < self.bar_range_list[i]:
                    return i + offset * height
            return height - 1 + offset * height
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ids[i][j] = get_p_id(data[i][j], j)
        return ids

    @classmethod
    def create_seq_bar_data_meta(cls, name, width, height, bar_range_list=None):
        return SeqBarDataMeta(name, width, height, bar_range_list)

    @classmethod
    def create_std_bar_data_meta(cls, name, width, height, std):
        bar_range_list = [-3 * std, -2.5 * std, -2*std]
        sub_bar_num = height - 6
        for i in range(1, sub_bar_num):
            bar_range_list.append(4 * std * i / sub_bar_num - 2 * std)
        bar_range_list.extend([2 * std, 2.5 * std, 3 * std])
        return cls.create_seq_bar_data_meta(name, width, height, bar_range_list)

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

        return cls.create_seq_bar_data_meta(name, width, height, bar_range_list)
