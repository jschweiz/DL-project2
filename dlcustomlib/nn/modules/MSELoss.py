from ..Loss import Loss

class MSELoss(Loss):
    def loss(self, predicted, labels):
        return ((predicted - labels)**2).mean()
    
    def dloss(self, predicted, labels):
        return 2 * (predicted - labels)/(len(predicted))