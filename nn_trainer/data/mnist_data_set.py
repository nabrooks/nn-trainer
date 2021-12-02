import torch
import numpy
from torch.utils.data.dataset import Dataset

from torchvision import datasets
from torchvision import transforms

class MnistDataSet(Dataset):
    def __init__(self, latent_size = 100, img_size = 28):
        self.latent_size = latent_size
        self.img_size = img_size
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5), std=(0.5)), transforms.Scale(img_size), transforms.Resize((img_size, img_size))])
        self.mnist_train_data = datasets.MNIST(root = 'data', train = True, transform = transform, download = True)

    def __len__(self):
        return int(len(self.mnist_train_data))

    def __getitem__(self, index):
        output, input = self.mnist_train_data.__getitem__(index)
        z = torch.randn(self.latent_size).numpy()
        r = numpy.concatenate(([input], z))
        return r, output.numpy()*0.5#0.5*(output.numpy() + 1)

    @property
    def input_shape(self):
        return self.__getitem__(0)[0].shape

    @property
    def output_shape(self):
        return self.__getitem__(0)[1].shape
