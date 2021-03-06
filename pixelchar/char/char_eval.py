from sklearn.metrics import roc_auc_score
import numpy as np
import sys


class CharEval(object):
    def __init__(self):
        self.cnt = 0

    def push(self, loss, predict, label, attach_data_list):
        self.cnt += 1

    def pop(self):
        self.cnt = 0


class LossCharEval(CharEval):
    def __init__(self, save_call=None):
        super(LossCharEval, self).__init__()
        self.loss = 0.0
        self.min_loss = sys.float_info.max
        self.save_call = save_call

    def push(self, loss, predict, label, attach_data_list):
        super(LossCharEval, self).push(loss, predict, label, attach_data_list)
        # 损失
        self.loss += loss

    def pop(self):
        loss = "loss: %.4f" % (self.loss / self.cnt)
        if (self.loss / self.cnt) < self.min_loss:
            self.min_loss = self.loss / self.cnt
            if self.save_call:
                self.save_call()
        self.loss = 0
        super(LossCharEval, self).pop()
        return loss


class AUCCharEval(CharEval):
    def __init__(self):
        super(AUCCharEval, self).__init__()
        self.label_list = []
        self.predict_list = []

    def push(self, loss, predict, label, attach_data_list):
        self.label_list.append(label)
        self.predict_list.append(predict)
        super(AUCCharEval, self).push(loss, predict, label, attach_data_list)

    def pop(self):
        auc = "auc:%.4f" % (roc_auc_score(np.concatenate(self.label_list, axis=0), np.concatenate(self.predict_list, axis=0)))
        self.label_list.clear()
        self.predict_list.clear()
        super(AUCCharEval, self).pop()
        return auc


class HitCharEval(CharEval):
    def __init__(self):
        super(HitCharEval, self).__init__()
        self.base_p = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.tp_fp_list = [0, 0, 0, 0, 0]
        self.tp_list = [0, 0, 0, 0, 0]

    def push(self, loss, predict, label, attach_data_list):
        super(HitCharEval, self).push(loss, predict, label, attach_data_list)
        # 精确率
        for p, l in zip(predict, label):
            for i in range(len(self.base_p)):
                if p > self.base_p[i]:
                    self.tp_fp_list[i] += 1
                    if l > 0.5:
                        self.tp_list[i] += 1

    def pop(self):
        hit_list = ["hit%.1f:%d/%d=%.4f" % (self.base_p[i], self.tp_list[i], self.tp_fp_list[i], self.tp_list[i] / (self.tp_fp_list[i] + 0.001)) for i in range(len(self.base_p))]
        for i in range(len(self.base_p)):
            self.tp_list[i] = 0
            self.tp_fp_list[i] = 0
        super(HitCharEval, self).pop()
        return '\t'.join(hit_list)


class PrecisionCharEval(CharEval):
    def __init__(self):
        super(PrecisionCharEval, self).__init__()
        self.threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.p_list = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.p_cnt_list = [0, 0, 0, 0, 0]

    def push(self, loss, predict, label, attach_data_list):
        super(PrecisionCharEval, self).push(loss, predict, label, attach_data_list)
        # 精确率
        for id, threshold in enumerate(self.threshold_list):
            tp_fp = 0
            tp = 0
            for p, l in zip(predict, label):
                if p > threshold:
                    tp_fp += 1
                    if l > 0.5:
                        tp += 1
            if tp_fp > 0:
                self.p_list[id] += tp / tp_fp
                self.p_cnt_list[id] += 1

    def pop(self):
        p_list = ["Precision%.1f:%.4f" % (self.threshold_list[i], self.p_list[i] / (self.p_cnt_list[i] + 0.0001)) for i in range(len(self.threshold_list))]
        for i in range(len(self.threshold_list)):
            self.p_list[i] = 0
            self.p_cnt_list[i] = 0
        super(PrecisionCharEval, self).pop()
        return '\t'.join(p_list)
