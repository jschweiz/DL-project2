from helpers import generate_data, evaluate_model
import torch.nn as nn_t, torch.optim as optim_t
import dlcustomlib.nn as nn_c, dlcustomlib.optim as optim_c
import matplotlib.pyplot as plt
import time, argparse

parser = argparse.ArgumentParser(description='Train and compare customlib and torch.')
parser.add_argument('--r', dest='r', default=10, type=int, help='Number of rounds on which the training is averaged')
parser.add_argument('--s', dest='s', default=1000, type=int, help='Number of training and testing datapoints')
parser.add_argument('--lr', dest='lr', default=0.01, type=float, help='Learning rate of SGD optimizer')
parser.add_argument('--wd', dest='wd',  default=0, type=float, help='Weight decay of SGD optimizer')
parser.add_argument('--e', dest='e',  default=1000, type=int, help='number of epochs')
parser.add_argument('--hl', dest='hl', default=25, type=int, help='Number of hidden layers')
args = parser.parse_args()


dict_models = {"Custom framework" : True, "Pytorch framework" : False}
datasets = (generate_data(args.s), generate_data(args.s))

def model_generator():
    nn = nn_c if CUSTOM else nn_t
    optim = optim_c if CUSTOM else optim_t
    net = nn.Sequential(nn.Linear(2,args.hl), nn.ReLU(), nn.Linear(args.hl,args.hl), nn.ReLU(), nn.Linear(args.hl,1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
    return net, criterion, optimizer

print("Running with parameters: {}".format(args))

plt.figure()
for key in dict_models.keys():

    print("Training {}".format(key))
    start = time.perf_counter()

    CUSTOM = dict_models[key]
    
    tr_mean, tr_std, te_mean, te_std, tr_loss = evaluate_model(datasets, model_generator,
    rounds = args.r, epochs=args.e, is_custom=dict_models[key])

    avg_time = (time.perf_counter() - start) / args.r
    plt.plot(tr_loss.detach(), label = "{}".format(key))
    print("\t - {}: \t training:  {:.3f} +/-  {:.3f} \t test: {:.3f} +/-  {:.3f} \t computed in {} s".format(
        key, tr_mean, tr_std, te_mean, te_std, avg_time))

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.savefig("data/plots/img_test_{}.jpeg".format(args))