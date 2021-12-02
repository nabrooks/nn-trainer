from argparse import ArgumentError
from typing import Tuple
from numpy import AxisError
from torch.utils.data.dataset import Dataset

from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import nn_trainer
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
args['num_epochs'] = 700
args['batch_size'] = 64
args['latent_dim'] = 0

z_size = 100
img_height_width = 32

data_set = nn_trainer.data.MnistDataSet(latent_size=100, img_size=img_height_width)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        self.onehot = torch.zeros(10, 10)
        self.onehot = self.onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)

        if torch.cuda.is_available():
            self.onehot = self.onehot.cuda()
            
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        
        z = input[:,1:].view(-1, 100, 1, 1)
        class_labels = input[:,0].long()
        label_one_hot = self.onehot[class_labels]#F.one_hot(class_labels, num_classes=10)
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(z)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label_one_hot)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, int(d/2), 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)
        self.fill = torch.zeros([10, 10, img_height_width, img_height_width])
        for i in range(10):
            self.fill[i, i, :, :] = 1
        
        if torch.cuda.is_available():
            self.fill = self.fill.cuda()

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, output):
        label = input[:,0].long()
        label = self.fill[label]

        x = F.leaky_relu(self.conv1_1(output), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.conv4(x).view(input.shape[0], 1)

        # x = torch.sigmoid(x)

        return x

# Generator factory method used to allow gan object to create generator
# lazily (only when needed). can be exchanged easily
def generator_ctor():
    G = generator()
    G.weight_init(0,0.02)
    #init_weights(G)
    return G

def discriminator_ctor():
    D = discriminator()
    D.weight_init(0,0.02)
    #init_weights(D)
    return D

netG = generator_ctor()
netD = discriminator_ctor()

trainer = WassersteinGanNnTrainer(generator_neural_network=netG, discriminator_neural_network=netD, verbose=True, max_epoch_count=args["num_epochs"], batch_size=args["batch_size"], output_dir=OUTPUT_PATH, logger=LOGGER)

# Create callbacks for logging and plotting state
callbacks = [ DataPlotterCallback(data_set, args["sample_size"], netG, MnistCganImageDataPlotter(), trainer.output_directory_path, logger=LOGGER) ]

# Gan instantiation and training here
trainer.train(data_set, data_set, callbacks=callbacks, metrics=[RMSE, MSE, MAE, AverageCosineSimilarity])

trainer.save_model()