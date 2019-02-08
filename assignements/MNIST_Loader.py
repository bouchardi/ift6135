from torch.utils.data import sampler, DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def load_dataset(batch_size, data_path='../data'):
    # ChunkSampler class is from https://github.com/pytorch/vision/issues/168
    class ChunkSampler(sampler.Sampler):
        """Samples elements sequentially from some offset.
        Arguments:
            num_samples: # of desired datapoints
            start: offset where we should start selecting from
        """

        def __init__(self, num_samples, start=0):
            self.num_samples = num_samples
            self.start = start

        def __iter__(self):
            return iter(range(self.start, self.start + self.num_samples))

        def __len__(self):
            return self.num_samples

    train_set = MNIST(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
    test_set = MNIST(root=data_path, train=False, transform=transforms.ToTensor(), download=True)

    train_set_size = len(train_set)
    NUM_TRAIN = int(0.8 * train_set_size)  # cast to int to avoid TypeError
    NUM_VAL = train_set_size - NUM_TRAIN
    NUM_TEST = len(test_set)

    trainset = DataLoader(train_set, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN, 0), shuffle=False)
    validset = DataLoader(train_set, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN), shuffle=False)
    testset = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Data Loading Complete!")

    return trainset, validset, testset