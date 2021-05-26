from helpers import evaluate_model, generate_description
from to_remove.dlc_practical_prologue import generate_pair_sets
import torch.nn as nn_t, torch.optim as optim_t
import dlcustomlib.nn as nn_c, dlcustomlib.optim as optim_c
import matplotlib.pyplot as plt
import time

# MODELS TO COMPARE 
ROUNDS = 5
EPOCHS= 10
NB_SAMPLE = 1000
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0

# NETWORK GENERATOR
def network_generator():
    nn = nn_c if CUSTOM else nn_t
    optim = optim_c if CUSTOM else optim_t
    net = nn.Sequential(
        nn.Conv2d(1, 4, 3), nn.ReLU(),  # 1x14x14 -> 6x12x12
        nn.Conv2d(4, 16, 3), nn.ReLU(),  # -> 16x10x10
        nn.Conv2d(16, 32, 3), nn.ReLU(),  # -> 32x8x8
        nn.Flatten(),
        nn.Linear(2048, 256), nn.ReLU(),
        nn.Linear(256, 25), nn.ReLU(),
        nn.Linear(25, 10)
        )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return net, criterion, optimizer

# DESCRIPTION VARIABLES
train_input_, train_target_, train_classes_, test_input_, test_target_, test_classes_ = generate_pair_sets(NB_SAMPLE)
train_input, train_target = train_input_[:,0].unsqueeze(1), train_classes_[:,0]
test_input, test_target = test_input_[:,0].unsqueeze(1), test_classes_[:,0]
datasets = (train_input, train_target), (test_input, test_target)
dict_models = {"Custom framework" : True,
 "Pytorch framework" : False}
CUSTOM = True
description_net = generate_description(network_generator)
description_params = 'r={},e={},s={},lr={},wd={}'.format(ROUNDS, EPOCHS, NB_SAMPLE, LEARNING_RATE, WEIGHT_DECAY)
print("Running with parameters: {} and model: {}".format(description_params, description_net))

# TRAINING SECTION
plt.figure()
for key in dict_models.keys():

    print("Training {}".format(key))
    start = time.perf_counter()

    CUSTOM = dict_models[key]
    
    tr_mean, tr_std, te_mean, te_std, tr_loss = evaluate_model(datasets, network_generator,
    rounds = ROUNDS, epochs=EPOCHS, is_custom=dict_models[key], show_progress=True)

    avg_time = (time.perf_counter() - start) / ROUNDS
    plt.plot(tr_loss.detach(), label = "{}".format(key))
    print("\t - {}: \t training:  {:.3f} +/-  {:.3f} \t test: {:.3f} +/-  {:.3f} \t computed in {} s".format(
        key, tr_mean, tr_std, te_mean, te_std, avg_time, show_progress=True))

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.savefig("data/mnist/img_{}_{}.jpeg".format(description_params, description_net))