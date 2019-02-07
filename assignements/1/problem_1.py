import numpy as np

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
#from sklearn.metrics import accuracy_score


# TODO:
# - Refactor forward and backward methods to generalize to L layers
# - Add an extra dimension to W and X for the bias!


class NN(object):

    def __init__(self,
                 input_size=784,
                 output_size=10,
                 hidden_layers_size=[512, 1024],
                 init='glorot',
                 lr=1.e-2):

        self.input_size = input_size
        self.hidden_layers_size = hidden_layers_size
        self.output_size = output_size
        self.lr = lr
        self.train = False
        self.init=init

        self._initialize_weights(init)

    def _initialize_weights(self, init_method):
        def _random(neurons_in, neurons_out):
            return np.random.normal(0, 0.01, (neurons_in, neurons_out))
        def _zeros(neurons_in, neurons_out):
            return np.zeros((neurons_in, neurons_out))
        def _normal(neurons_in, neurons_out):
            return np.random.normal(0, 1, (neurons_in, neurons_out))
        def _glorot(neurons_in, neurons_out,low,high):
            return np.random.uniform(low,high,(neurons_in, neurons_out))

        # Not super clean...
        init_weights = {
                'random': _random,
                'zeros': _zeros,
                'normal':_normal,
                'glorot': _glorot
                }

        sizes = [self.input_size] + self.hidden_layers_size + [self.output_size]
        if (init_method =='random') or (init_method =='zeros') or (init_method == 'normal'):
            self.W = [init_weights[init_method](sizes[i], sizes[i+1]) for i in range(len(sizes) - 1 )]
        else:
            self.W = [init_weights[init_method](sizes[i], sizes[i+1], -np.sqrt(6/(sizes[i]+sizes[i+1])),np.sqrt(6/(sizes[i]+sizes[i+1]))) for i in range(len(sizes) - 1 )]

        self.b = [np.zeros((1, neurons)) for neurons in sizes[1:]]

    def _add_bias(self, h, W, b):
        h = np.concatenate([h, np.ones((1, h.shape[1]))], axis=0)
        W = np.concatenate([W, b])
        return h, W

    def forward(self, X):
        h = X
        cache = [(h, None)]
        for W, b in zip(self.W, self.b):
            # Add bias
            hb, Wb = self._add_bias(h, W, b)
            a = np.dot(Wb.T, hb)
            h = self._sigmoid(a)
            cache.append((h, a))
        return h, cache

    def _sigmoid(self, X):
         return 1 / (1 + np.exp(-X))

    def _sigmoid_deriv(self, X):
        return self._sigmoid(X) * (1 - self._sigmoid(X))

    def _softmax(self, X):
        """ Softmax activation function """
        e_x = np.exp(X - np.max(X))
        return e_x / e_x.sum(axis=0)

    def loss(self, prediction, target):
        """ Cross-entropy loss """
        return -np.log((prediction * target).sum(axis=0)).mean()

    def backward(self, target, prediction, cache):
        grads_W = []
        grad_a = - (target - prediction)
        for i in range(len(cache) - 1):
            index = len(cache) - i - 2
            grad_W = np.dot(grad_a, cache[index][0].T)
            grad_b = grad_a
            if index:
                grad_h = np.dot(self.W[index], grad_a)
                grad_a = np.multiply(grad_h, self._sigmoid_deriv(cache[index][1]))
            grads_W.append(grad_W.T)
        return [g for g in reversed(grads_W)]

    def update_weights(self, grads):
        if not self.train:
            raise Exception('You should not update weights while validating/testing')
        self.W = [self.W[i] - (self.lr * grads[i]) for i in range(len(self.W))]

    def training(self):
        self.train = True

    def eval(self):
        self.train = False


def load_dataset(batch_size, data_path='./data'):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = DataLoader(dataset=datasets.MNIST(root=data_path,
                                                 train=True,
                                                 download=True,
                                                 transform=transform),
                          batch_size=batch_size,
                          shuffle=True)
    testset = DataLoader(dataset=datasets.MNIST(root=data_path,
                                                train=False,
                                                download=True,
                                                transform=transform),
                         batch_size=batch_size,
                         shuffle=True)
    return trainset, testset


def preprocess(batch, n_class=10):
    """
    Transform model_input in flat vector and target in one-hot encoded
    """
    model_input, target = batch
    model_input = model_input.numpy().reshape((model_input.shape[0], -1)).T
    target_one_hot = np.zeros((n_class, target.shape[0]))
    target_one_hot[target, np.arange(target.shape[0])] = 1
    return model_input, target_one_hot


def main(model, trainset, validset, epochs):
    loss_vector = np.zeros([epochs,1])
    acc=np.zeros([epochs,1])

    for epoch in range(epochs):
        acc_=0

        # Training
        loss = 0
        model.training()
        for i, batch in enumerate(trainset):
            model_input, target = preprocess(batch)
            prediction, cache = model.forward(model_input)
            grads = model.backward(target, prediction, cache)
            model.update_weights(grads)
            loss += model.loss(prediction, target)
        loss_vector[epoch,0] = loss/(i+1)
        print(f'Train loss={loss / (i + 1)} at epoch {epoch}')

        # Validation
        loss = 0
        model.eval()
        for i, batch in enumerate(validset):
            model_input, target = preprocess(batch)
            prediction, _ = model.forward(model_input)
            loss += model.loss(prediction, target)
            targ = np.argmax(np.asarray(target), axis=0)
            pred = np.argmax(np.asarray(prediction), axis=0)
            for j in range(len(target[1])):
                if targ[j] == pred[j]:
                    acc_ = acc_+1
        acc[epoch,0]=(acc_/len(validset))
        print(acc)
        print(f'Valid loss={loss / (i + 1)} at epoch {epoch}')
    #init_=model.init
    #np.savetxt(init_+'.txt', loss_vector, delimiter=',', fmt='%10.5f')

if __name__ == '__main__':
    epochs =13
    batch_size = 128

    model = NN()
    trainset, validset = load_dataset(batch_size)


    main(model=model,
         trainset=trainset,
         validset=validset,
         epochs=epochs)

