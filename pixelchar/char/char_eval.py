from sklearn.metrics import roc_auc_score


class CharEval(object):
    def __init__(self):
        self.cnt = 0

    def push(self, loss, predict, label):
        self.cnt += 1

    def pop(self):
        self.cnt = 0


class LossCharEval(CharEval):
    def __init__(self):
        super(LossCharEval, self).__init__()
        self.loss = 0.0

    def push(self, loss, predict, label):
        super(LossCharEval, self).push(loss, predict, label)
        # 损失
        self.loss += loss

    def pop(self):
        loss = "loss: %.4f" % (self.loss / self.cnt)
        self.loss = 0
        super(LossCharEval, self).pop()
        return loss


class AUCCharEval(CharEval):
    def __init__(self):
        super(AUCCharEval, self).__init__()
        self.auc = 0.0

    def push(self, loss, predict, label):
        super(AUCCharEval, self).push(loss, predict, label)
        self.auc += roc_auc_score(label, predict)

    def pop(self):
        auc = "auc:%.4f" % (self.auc / self.cnt)
        self.auc = 0.0
        super(AUCCharEval, self).pop()
        return auc


class PrecisionCharEval(CharEval):
    def __init__(self):
        super(PrecisionCharEval, self).__init__()
        self.base_p = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.tp_fp_list = [0, 0, 0, 0, 0]
        self.tp_list = [0, 0, 0, 0, 0]

    def push(self, loss, predict, label):
        super(PrecisionCharEval, self).push(loss, predict, label)
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
        super(PrecisionCharEval, self).pop()
        return '\t'.join(hit_list)
