from torch.utils.data import sampler, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

def load_dataset(batch_size, data_path='/project/data'):
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

    train_set = ImageFolder(root=f'{data_path}/cat_dog_data/train', transform=transforms.ToTensor())
    valid_set = ImageFolder(root=f'{data_path}/cat_dog_data/val', transform=transforms.ToTensor())

    trainset = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validset = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

    print("Data Loading Complete!")

    return trainset, validset, None
