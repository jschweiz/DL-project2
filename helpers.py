import torch, math

def compute_error_ratio(predicted, labels):
    e = 0
    if predicted.shape != labels.shape:
        (predicted.argmax(axis=1) != labels)*1/predicted.shape[0]
    else:
        e = 1 - (torch.round(predicted) == labels).float().mean()
    return e


def generate_data(nb):
    train_input = torch.rand(nb,2)
    train_target = torch.empty(nb, dtype=torch.long)
    train_target = train_input.sub(0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).float().view(-1,1)
    return train_input, train_target


def train_model(model, optimizer, criterion, train_input, train_target, mini_batch_size=100, epochs=100, is_custom=True, log_loss=False):
    tr_losses, te_losses = torch.empty((epochs,1)), torch.empty((epochs,1)) #Training loss over epochs

    for epoch in range(epochs):
        for inputs, labels in zip(train_input.split(mini_batch_size), train_target.split(mini_batch_size)):

            # reset optimizer, forward and compute loss 
            optimizer.zero_grad()
            predicted = model(inputs)
            loss = criterion(predicted, labels)

            # calculate gradients (using backward on the model for custom and autograd.backward for torch)
            if is_custom:
                dloss = criterion.dloss(predicted, labels)
                model.backward(dloss)
            else:
                loss.backward()
            optimizer.step()
            
            #err = compute_error_ratio(predicted, labels)
            tr_losses[epoch] += loss
        
        if log_loss and epoch%100==0: print("Epoch: {}\t Loss={:.3f}".format(epoch, tr_losses[epoch].numpy()[0]))
    return tr_losses, te_losses



def evaluate_model(datasets, model_generator, rounds=10, epochs = 10, is_custom = True, log_loss=False):

    (train_input, train_target), (test_input, test_target) = datasets

    tr_error, te_error = torch.empty((rounds,1)), torch.empty((rounds,1))
    tr_losses, te_losses = torch.empty((rounds, epochs)), torch.empty((rounds, epochs))
    
    for r in range(rounds):
        
        # Generate model and training/test data
        model, criterion, optimizer = model_generator()
        torch.set_grad_enabled(not is_custom)
        
        # Train model
        tr_loss, te_loss = train_model(model, optimizer, criterion, train_input, train_target, epochs=epochs, is_custom=is_custom, log_loss=log_loss)
        predicted_test = model(test_input)
        predicted_train = model(train_input)
        
        # Compute error rate
        err_test = compute_error_ratio(predicted_test, test_target)
        err_train = compute_error_ratio(predicted_train, train_target)
        
        # Save all data
        tr_losses[r], te_losses[r] = tr_loss.t(), te_loss.t()
        tr_error[r], te_error[r] = err_train, err_test
        
    return torch.mean(tr_error), torch.std(tr_error), torch.mean(te_error), torch.std(te_error), torch.mean(tr_losses, axis = 0) 