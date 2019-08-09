class CharEval(object):
    def __init__(self):
        self.cnt = 0

    def push(self, loss, predict, label):
        self.cnt += 1

    def pop(self):
        self.cnt = 0
