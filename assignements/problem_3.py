import numpy as np
import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import CatsAndDogs_Loader


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(16 * 16 * 64, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        flatten = out.view(out.size(0), -1)
        fc = self.fc(flatten)
        return fc

def get_accuracy(target, prediction):
    res = np.argmax(target.detach().numpy(), axis=1) == np.argmax(prediction.detach().numpy(), axis=1)
    return len(res[res]) / len(res)

def main(model, trainset, validset, testset, epochs, learning_rate, device):

    # loss function
    criterion = nn.BCELoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model = model.to(device)

    train_errors = []
    eval_errors = []

    softmax = nn.Softmax(dim=1)

    print("Training begins...")
    for epoch in range(epochs):

        # print eval_error after each epoch
        print('\nEpoch {}'.format(epoch))

        train_error = 0
        train_num = 0

        # model in train mode
        model.train()

        for model_input, labels in trainset:

            model_input = model_input.to(device)
            # one hot encode
            onehot = torch.zeros((labels.shape[0], 2))
            onehot[np.arange(onehot.shape[0]), labels] = 1
            onehot = onehot.to(device)

            # forward pass
            outputs = model(model_input)
            # loss function
            loss = criterion(softmax(outputs), onehot)
            # zero gradient buffer
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # gradient descent step
            optimizer.step()

            # add the loss
            train_error += loss.item()
            train_num += 1
        print('\tTrain error: {:.4f}'.format(train_error / train_num))
        eval_error = 0
        eval_num = 0
        accuracy = 0

        # model in eval mode
        model.eval()

        for model_input, labels in validset:
            model_input = model_input.to(device)
            # one hot encode
            onehot = torch.zeros((labels.shape[0], 2))
            onehot[np.arange(onehot.shape[0]), labels] = 1
            onehot = onehot.to(device)

            # forward pass
            outputs = model(model_input)

            # loss function
            loss = criterion(softmax(outputs), onehot)
            accuracy += get_accuracy(outputs, onehot)

            # add the loss
            eval_error += loss.item()
            eval_num += 1

        # save the loss
        train_errors.append(train_error / train_num)
        eval_errors.append(eval_error / eval_num)

        print('\tAccuracy: {:.4f}'.format(accuracy / eval_num))
        print('\tEval error: {:.4f}'.format(eval_error / eval_num))

    # plot train and validation loss
    plot_loss(train_error,eval_error)

    #Testing the model
    print("Testing begins...")
    # model in eval mode
    model.eval()

    correct = 0
    total = 0

if __name__ == '__main__':
    epochs = 10
    batch_size = 256
    lr = 1.e-5

    model = CNN()
    print(model)

    print("\n\n# Parameters: ", sum([param.nelement() for param in model.parameters()]))
    trainset, validset, testset = CatsAndDogs_Loader.load_dataset(batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(model=model,
         trainset=trainset,
         validset=validset,
         testset=testset,
         epochs=epochs,
         learning_rate=lr,
         device=device)
