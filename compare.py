from helpers import generate_data, evaluate_model, generate_description
import torch.nn as nn_t, torch.optim as optim_t
import dlcustomlib.nn as nn_c, dlcustomlib.optim as optim_c
import matplotlib.pyplot as plt
import time, argparse

## Parser section
parser = argparse.ArgumentParser(description='Train and compare customlib and torch.')
parser.add_argument('--r', dest='r', default=10, type=int, help='Number of rounds on which the training is averaged')
parser.add_argument('--s', dest='s', default=1000, type=int, help='Number of training and testing datapoints')
parser.add_argument('--lr', dest='lr', default=0.01, type=float, help='Learning rate of SGD optimizer')
parser.add_argument('--wd', dest='wd',  default=0, type=float, help='Weight decay of SGD optimizer')
parser.add_argument('--e', dest='e',  default=1000, type=int, help='number of epochs')
parser.add_argument('--hl', dest='hl', default=25, type=int, help='Number of hidden layers')
args = parser.parse_args()

# NETWORK GENERATOR
def network_generator():
    nn = nn_c if CUSTOM else nn_t
    optim = optim_c if CUSTOM else optim_t
    net = nn.Sequential(nn.Linear(2,args.hl), nn.ReLU(), nn.Linear(args.hl,args.hl), nn.ReLU(), nn.Linear(args.hl,1))
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
    return net, criterion, optimizer

# DESCRIPTION VARIABLES
dict_models = {"Pytorch framework" : False, "Custom framework" : True}
datasets = (generate_data(args.s), generate_data(args.s))
CUSTOM = True
description_net = generate_description(network_generator)
description_params = str(args.__dict__).replace(' ','').replace('\'','').replace('{','').replace('}','').replace(':','=')
print("Running with parameters: {} and model: {}".format(description_params, description_net))

# TRAINING SECTION
plt.figure()
for key in dict_models.keys():

    print("Training {}".format(key))
    start = time.perf_counter()

    CUSTOM = dict_models[key]
    
    tr_mean, tr_std, te_mean, te_std, tr_loss = evaluate_model(datasets, network_generator,
    rounds = args.r, epochs=args.e, is_custom=dict_models[key], show_progress=True)

    avg_time = (time.perf_counter() - start) / args.r
    plt.plot(tr_loss.detach(), label = "{}".format(key))
    print("RESULTS => {}: \t training:  {:.3f} +/-  {:.3f} \t test: {:.3f} +/-  {:.3f} \t computed in {} s\n".format(
        key, tr_mean, tr_std, te_mean, te_std, avg_time))
    
# PLOT SETTINGS
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.savefig("data/report/plot_{}_{}.jpeg".format(description_params, description_net))