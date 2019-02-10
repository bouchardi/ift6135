import numpy as np
import argparse

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt

import MNIST_Loader


# TODO:
# - Plot check_grads?!
# - Divide grads and loss by batch_size?
# - Test, valid set same same?


class NN(object):

    def __init__(self,
                 input_size=784,
                 output_size=10,
                 hidden_layers_size=[512, 1024],
                 init='zeros',
                 activation='sigmoid',
                 lr=0.1):

        self.input_size = input_size
        self.hidden_layers_size = hidden_layers_size
        self.output_size = output_size
        self.lr = lr
        self.train = False
        self.init=init

        self._initialize_weights(init)
        self._initialize_activation(activation)

    def _initialize_weights(self, init_method):
        def _random(neurons_in, neurons_out):
            return np.random.normal(0, 0.01, (neurons_in, neurons_out))
        def _zeros(neurons_in, neurons_out):
            return np.zeros((neurons_in, neurons_out))
        def _normal(neurons_in, neurons_out):
            return np.random.normal(0, 1, (neurons_in, neurons_out))
        def _glorot(neurons_in, neurons_out):
            low = -np.sqrt(6 / (neurons_in + neurons_out))
            high = np.sqrt(6 / (neurons_in + neurons_out))
            return np.random.uniform(low, high, (neurons_in, neurons_out))

        init_weights = {
                'random': _random,
                'zeros': _zeros,
                'normal':_normal,
                'glorot': _glorot
                }

        sizes = [self.input_size] + self.hidden_layers_size + [self.output_size]
        self.W = [init_weights[init_method](sizes[i], sizes[i+1]) for i in range(len(sizes) - 1 )]
        self.b = [np.zeros((1, neurons)) for neurons in sizes[1:]]

    def _initialize_activation(self, activation):
        if activation == 'sigmoid':
            self.activation_f, self.activation_deriv_f = self._sigmoid, self._sigmoid_deriv
        elif activation == 'tanh':
            self.activation_f, self.activation_deriv_f = self._tanh, self._tanh_deriv
        elif activation == 'linear':
            self.activation_f, self.activation_deriv_f = self._linear, self._linear_deriv

    def _sigmoid(self, X):
         return 1 / (1 + np.exp(-X))

    def _sigmoid_deriv(self, X):
        return self._sigmoid(X) * (1 - self._sigmoid(X))

    def _tanh(self, X):
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    def _tanh_deriv(self, X):
        return 1 - self._tanh(X)**2

    def _linear(self, X):
        return X

    def _linear_deriv(self, X):
        return 1

    def _add_bias(self, h, W, b):
        h = np.concatenate([h, np.ones((1, h.shape[1]))], axis=0)
        W = np.concatenate([W, b])
        return h, W

    def forward(self, X, model_W=None):
        model_W = self.W if model_W is None else model_W
        h = X
        cache = [(h, None)]
        for i, (W, b) in enumerate(zip(model_W, self.b)):
            # Add bias
            hb, Wb = self._add_bias(h, W, b)
            a = np.dot(Wb.T, hb)
            # Different activation function for last layer (softmax)
            if i == len(model_W) - 1:
                h = self._softmax(a)
            else:
                h = self.activation_f(a)
            cache.append((h, a))
        return h, cache

    def _softmax(self, X):
        """ Softmax activation function """
        e_x = np.exp(X - np.max(X))
        return e_x / e_x.sum(axis=0)

    def loss(self, prediction, target):
        """ Cross-entropy loss """
        return -np.log((prediction * target).sum(axis=0)).mean()

    def backward(self, target, prediction, cache):
        grads = []
        grad_a = - (target - prediction)
        for i in range(len(cache) - 1):
            index = len(cache) - i - 2
            grad_W = np.dot(grad_a, cache[index][0].T)
            grad_b = grad_a
            if index:
                grad_h = np.dot(self.W[index], grad_a)
                grad_a = np.multiply(grad_h, self.activation_deriv_f(cache[index][1]))
            grads.append((grad_W.T, np.sum(grad_b.T, axis=0)))
        return [g for g in reversed(grads)]

    def update_weights(self, grads):
        if not self.train:
            raise Exception('You should not update weights while validating/testing')
        self.W = [self.W[i] - (self.lr * grads[i][0]) for i in range(len(self.W))]
        self.b = [self.b[i] - (self.lr * grads[i][1]) for i in range(len(self.W))]

    def training(self):
        self.train = True

    def eval(self):
        self.train = False


def check_grads(model, batch, p=1):
    model_input, target = preprocess(batch)
    # Keep only one example
    model_input = model_input[:, :1]
    target = target[:, :1]
    # Get the grads
    prediction, cache = model.forward(model_input)
    grads = model.backward(target, prediction, cache)
    # Get the numerical approximation of grads for p values for different values of N
    diff = []
    legends = []
    for k in range(5):
        N = 10**k
        # Get the numerical app gradients
        num_grads = get_numerical_grads(model_input, target, model, N, p)
        # Compare the numerical and the 'real' gradients
        diff.append(np.max(abs(grads[2][0][:p, :model.W[2].shape[1]] - num_grads[:p, :model.W[2].shape[1]])))
        legends.append(f'N = {N}')
    # Plot the difference
    plt.plot(diff)
    plt.legend(legends)
    plt.show()


def get_numerical_grads(X, y, model, N, p):
    num_grad = np.zeros(model.W[2].shape)
    perturb = np.zeros(model.W[2].shape)
    e = 1 / N
    for i in range(p):
        for j in range(model.W[2].shape[1]):
            perturb[i, j] = e
            W = model.W.copy()
            W[2] += perturb
            loss2 = model.loss(model.forward(X, model_W=W)[0], y)
            W[2] -= 2 * perturb
            loss1 = model.loss(model.forward(X, model_W=W)[0], y)
            num_grad[i, j] = (loss2 - loss1) / (2 * e)
            perturb[i, j] = 0
    return num_grad


def get_accuracy(target, prediction):
    res = np.argmax(target, axis=0) == np.argmax(prediction, axis=0)
    return len(res[res]) / len(res)


def train(model, trainset, validset, testset, epochs, check_grad=False):
    loss_vector = np.zeros([epochs, 1])
    patience = 0
    best_accuracy = 0
    best_W = None
    best_b = None

    for epoch in range(epochs):
        # Training
        loss = 0
        model.training()
        for i, batch in enumerate(trainset):
            model_input, target = preprocess(batch)
            prediction, cache = model.forward(model_input)
            grads = model.backward(target, prediction, cache)
            model.update_weights(grads)
            loss += model.loss(prediction, target)
        loss_vector[epoch, 0] = loss / (i+1)
        print(f'Train loss={loss / (i + 1)} at epoch {epoch}')

        # Validation
        loss = 0
        accuracy = 0
        model.eval()
        for i, batch in enumerate(validset):
            model_input, target = preprocess(batch)
            prediction, _ = model.forward(model_input)
            loss += model.loss(prediction, target)
            accuracy += get_accuracy(target, prediction)
        print(f'Valid loss={loss / (i + 1)} at epoch {epoch}')
        print(f'Valid accuracy={accuracy / (i+1)} at epoch {epoch}')

        if accuracy / (i + 1) > best_accuracy:
            best_accuracy = accuracy / (i + 1)
            best_W = model.W.copy()
            best_b = model.b.copy()
            patience = 0
        else:
            patience += 1

        if patience > 2:
            break

    if check_grad:
        for batch in trainset:
            check_grads(model, batch)
            break

    model.W = best_W
    model.b = best_b

    test_accuracy = 0
    model.eval()
    for j, batch in enumerate(testset):
        model_input, target = preprocess(batch)
        prediction, _ = model.forward(model_input)
        test_accuracy += get_accuracy(target, prediction)
    print(f'Test accuracy={test_accuracy / (j+1)}')

    return loss_vector, accuracy / (i+1), test_accuracy / (j+1)


def preprocess(batch, n_class=10):
    """
    Transform model_input in flat vector and target in one-hot encoded
    """
    model_input, target = batch
    model_input = model_input.numpy().reshape((model_input.shape[0], -1)).T
    target_one_hot = np.zeros((n_class, target.shape[0]))
    target_one_hot[target, np.arange(target.shape[0])] = 1
    return model_input, target_one_hot


def plot_loss(loss_vector, epochs, legend):
    t = np.arange(loss_vector.size)
    plt.ylabel('Average Loss')
    plt.xlabel('Epoch')
    plt.title('Initialization Effect')
    plt.grid(True)
    plt.xlim(0, epochs)
    plt.plot(t, loss_vector)
    plt.legend(legend, loc='upper right')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1.e-2, type=float)
    parser.add_argument('--activation', default='sigmoid')
    parser.add_argument('--init', default='glorot')
    parser.add_argument('--h1', default=512, type=int)
    parser.add_argument('--h2', default=1024, type=int)
    args = parser.parse_args()

    init_methods = ['zeros', 'normal', 'glorot'] if args.init == 'all' else [args.init]

    trainset, validset, testset = MNIST_Loader.load_dataset(args.batch_size)

    for init in init_methods:
        model = NN(hidden_layers_size=[args.h1, args.h2],
                   init=init,
                   activation=args.activation,
                   lr=args.lr)

        results = train(model=model,
                        trainset=trainset,
                        validset=validset,
                        testset=validset,
                        epochs=args.epochs)
        loss_vector, valid_accuracy, test_accuracy = results
        with open('hp_results.txt', 'a') as text_file:
            text_file.write(f'{args.epochs} {args.batch_size} {args.lr} {args.activation} {args.init} {args.h1} {args.h2} {valid_accuracy} {test_accuracy}\n')

    if args.init == 'all':
        plot_loss(loss_vector, epochs=args.epochs, legend=init_methods)
