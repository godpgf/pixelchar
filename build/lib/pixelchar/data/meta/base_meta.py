import pickle


class DataMeta(object):
    def __init__(self, name):
        self.name = name

    @classmethod
    def save(cls, data_meta, path):
        data_meta_dict = {data_meta.name: data_meta for data_meta in data_meta} if isinstance(data_meta, list) else data_meta
        with open(path, "wb") as f:
            pickle.dump(data_meta_dict, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


class ValueDataMeta(DataMeta):
    def __init__(self, name, value_type='float32'):
        super(ValueDataMeta, self).__init__(name)
        self.value_type = value_type

    @classmethod
    def create_value_data_meta(cls, name, value_type='float32'):
        return ValueDataMeta(name, value_type)


class SeqDataMeta(DataMeta):
    def __init__(self, name, seq_length, max_item_size):
        super(SeqDataMeta, self).__init__(name)
        self.seq_length = seq_length
        self.max_item_size = max_item_size

    @classmethod
    def create_seq_data_meta(cls, name, seq_length, max_item_size):
        return SeqDataMeta(name, seq_length, max_item_size)
