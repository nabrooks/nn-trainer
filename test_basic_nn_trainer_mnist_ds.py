from argparse import ArgumentError
from typing import Tuple
from numpy import AxisError
from torch.utils.data.dataset import Dataset

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor


from nn_trainer.utils import *
import torch
import torch.optim
from nn_trainer.data import *
from nn_trainer.models import *
from nn_trainer.trainers import *
from nn_trainer.plotters import *
from nn_trainer.metrics import *
from nn_trainer.callbacks import *


args = parse_arguments()
MODEL_SIZE = 16
INPUT_SIZE = 64
LOGGER = Logger("nn_trainer")

args['training_subset_ratio'] = .5

args['model_size'] = MODEL_SIZE
args['input_size'] = INPUT_SIZE
args['num_epochs'] = 100
args['batch_size'] = 256*2
args['latent_dim'] = 0
args['learning_rate'] = 0.0005
args['feature_range_min'] = -1
args['feature_range_max'] = 1

class MnistDataSet(Dataset):
    def __init__(self):
        self.transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize((28, 28)) ])
        self.mnist_train_data = datasets.MNIST(root = 'data', train = True, transform = self.transform, download = True)

    def __len__(self):
        return int(len(self.mnist_train_data))

    def __getitem__(self, index):
        output, input = self.mnist_train_data.__getitem__(index)
        z = torch.randn(101).numpy()
        z[0] = float(input)
        return z, output.numpy()

    @property
    def input_shape(self):
        return self.__getitem__(0)[0].shape

    @property
    def output_shape(self):
        return self.__getitem__(0)[1].shape

data_set = MnistDataSet()
first_sample = data_set.__getitem__(0)

i = first_sample[0]
o = first_sample[1]

fake_labels = torch.LongTensor(np.random.randint(0, 10, args['batch_size']))
label_emb = torch.nn.Embedding(10, 10)
c = label_emb(fake_labels)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, labels):
        z = labels[:,1:]
        labels = labels[:,0]
        z = z.view(z.size(0), 100)
        labels = labels.long()
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, labels, x):
        x = x.view(x.size(0), 784)
        condition = labels[:,0].squeeze().long()
        c = self.label_emb(condition)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out

# Generator factory method used to allow gan object to create generator
# lazily (only when needed). can be exchanged easily
def generator_ctor():
    gNet = Generator()
    init_weights(gNet)
    return gNet

def discriminator_ctor():
    dNet = Discriminator()
    init_weights(dNet)
    return dNet

netG = generator_ctor()
netD = discriminator_ctor()

trainer = BasicNnTrainer(neural_network=netG, verbose=True, max_epoch_count=args["num_epochs"],batch_size=args["batch_size"], dtype=torch.float64, output_dir=OUTPUT_PATH, logger=LOGGER)


# Create callbacks for logging and plotting state
callbacks = [ 
    DataPlotterCallback(data_set, args["sample_size"], netG, MnistCganImageDataPlotter(), trainer.output_directory_path, logger=LOGGER) 
    ]

# Gan instantiation and training here
trainer.train(data_set, data_set, callbacks=callbacks, metrics=[RMSE, MSE, MAE, AverageCosineSimilarity])

paths = trainer.save_models()

for path in paths:
    LOGGER.info(paths)