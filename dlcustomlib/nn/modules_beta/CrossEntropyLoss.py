from ..Loss import Loss
import torch

def softmax(x):
        a,_ = torch.max(x, dim = 1)
        expo = torch.exp(x - a.view(-1,1))
        return expo / torch.sum(expo, dim = 1).view(-1,1)

def one_hot(target):
    res = torch.zeros((target.shape[0], 10))
    for i in range(target.shape[0]):
        res[i,target[i]] = 1
    return res

class CrossEntropyLoss(Loss):
    def loss(self,predicted, targets):
        rtargets = torch.arange(targets.shape[0])*predicted[0].shape[0]+targets
        soft = torch.log(softmax(predicted))
        inter = torch.take(soft,rtargets)
        loss = torch.sum(inter) * (-1/targets.shape[0])
        return  loss
    
    def dloss(self,predicted, targets):
        dloss = softmax(predicted) - one_hot(targets)
        return dloss/len(predicted)