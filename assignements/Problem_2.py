import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import MNIST_Loader


# TODO:
# - check that mnist data isn't in order from 1-9
# - I did train 80%, validate 20% (check MLP)
# - going to have to change the activation function in MLP from sigmoid (11% accuracy aka random) to ReLU (95% accuracy)
# - try using Leaky ReLU as activation function
# - can we use dropout?
# - add dilation?
# - check if its better to have smaller filters at the beginning or end of network (see textbook)
# - can we add batchnorm?
# - add more parameters to match MLP
# - separate train from evaluate (commented out right now)
# - can we change batch size? batch size 34 gives 98%


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d())

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d())

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d())

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d())

        self.fc = nn.Linear(2 * 2 * 256, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        flatten = out.view(out.size(0), -1)
        fc = self.fc(flatten)
        return fc

def plot_loss(train_errors, eval_errors):
    # Plot training and validation curve
    x = list(range(len(train_errors)))
    plt.plot(x, train_errors, 'm', label='Train')
    plt.plot(x, eval_errors, 'g', label='Validation')

    plt.xlabel('Number Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best', shadow=True, fancybox=True)
    plt.title("Loss")

    plt.show()


def main(model, trainset, validset, testset, epochs, learning_rate):

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model.load_state_dict(init_model_wts)

    train_errors = []
    eval_errors = []

    print("Training begins...")
    for epoch in range(epochs):

        train_error = 0
        train_num = 0

        # model in train mode
        model.train()

        for digits, labels in trainset:

            # zero gradient buffer
            optimizer.zero_grad()

            # forward pass
            outputs = model(digits)

            # loss function
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()

            # gradient descent step
            optimizer.step()

            # add the loss
            train_error += loss.item()
            train_num += 1

        eval_error = 0
        eval_num = 0

        # model in eval mode
        model.eval()

        for digits, labels in validset:

            # forward pass
            outputs = model(digits)

            # loss function
            loss = criterion(outputs, labels)

            # add the loss
            eval_error += loss.item()
            eval_num += 1

        # save the loss
        train_errors.append(train_error / train_num)
        eval_errors.append(eval_error / eval_num)

        # print eval_error after each epoch
        print('\nEpoch {}'.format(epoch + 1))
        print('\tTrain error: {:.4f}'.format(train_error / train_num))
        print('\tEval error: {:.4f}'.format(eval_error / eval_num))

    # plot train and validation loss
    plot_loss(train_error,eval_error)

    #Testing the model
    print("Testing begins...")
    # model in eval mode
    model.eval()

    correct = 0
    total = 0

    for digits, labels in testset:

        # forward pass
        outputs = model(digits)
        _, predicted = torch.max(outputs.data, 1)

        # save the accuary
        total += labels.size(0)
        correct += torch.sum(predicted == labels.data)

    print('Accuracy on the test set: {:.2f}%'.format(100 * correct / total))


if __name__ == '__main__':
    epochs =10
    batch_size = 128
    lr = 1.e-2

    model = CNN()
    print(model)

    print("\n\n# Parameters: ", sum([param.nelement() for param in model.parameters()]))

    # Save the initial weights of model
    init_model_wts = copy.deepcopy(model.state_dict())

    trainset, validset, testset = MNIST_Loader.load_dataset(batch_size)


    main(model=model,
         trainset=trainset,
         validset=validset,
         testset=testset,
         epochs=epochs,
         learning_rate=lr)

