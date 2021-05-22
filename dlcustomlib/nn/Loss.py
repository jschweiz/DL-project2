class Loss(object):
    def loss(self, predicted, labels):
        raise NotImplementedError
    def __call__(self, predicted, labels):
        return self.loss(predicted, labels)
    def dloss(self, predicted, target):
        raise NotImplementedError