from helpers import generate_data, evaluate_model
import dlcustomlib.nn as nn, dlcustomlib.optim as optim
import matplotlib.pyplot as plt
import torch

torch.set_grad_enabled(False)

# Generates a training and a test set of 1,000 points sampled uniformly in [0,1],
# each  with  alabel 0 if outside the disk centered at (0.5,0.5) of radius 1/√2π, and 1 inside,2 of  3
datasets = (generate_data(1000), generate_data(1000))

# Duilds a network with two input units, one output unit, three hidden layers of 25 units
def model_generator():
    net = nn.Sequential(
        nn.Linear(2,25), nn.ReLU(), # INPUT 2->25
        nn.Linear(25,25), nn.ReLU(), # HIDDEN 25->25
        nn.Linear(25,25), nn.ReLU(), # HIDDEN 25->25
        nn.Linear(25,25), nn.ReLU(), # HIDDEN 25->25
        nn.Linear(25,1)) # OUTPUT 25->1
    return net, nn.MSELoss(), optim.SGD(net.parameters())


# Trains it with MSE, logging the loss
tr_mean, tr_std, te_mean, te_std, tr_loss = evaluate_model(datasets, model_generator,
    rounds = 1, epochs=1000, is_custom=True, log_loss=True)

# Computes and prints the final train and the test errors
print("Results: \t training error_rate= {:.3f}\t test error_rate= {:.3f}".format(
    tr_mean, te_mean))
