from argparse import ArgumentError
from typing import Tuple
from numpy import AxisError

from torch.utils.data.dataloader import DataLoader
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
args['batch_size'] = 128
args['latent_dim'] = 0
args['learning_rate'] = 0.0005
args['feature_range_min'] = -1
args['feature_range_max'] = 1

class LinearGenerator(nn.Module):
    def __init__(self, in_samples:int, out_samples:int, hidden_layer_count:int = 1, hidden_sample_count:int = 15):
        super(LinearGenerator, self).__init__()

        self.fc_l0 = nn.Linear(in_samples, hidden_sample_count)
        self.bn0 = nn.BatchNorm1d(hidden_sample_count)

        hidden_layer_count = hidden_layer_count
        self.hidden_layers = nn.ModuleList()
        self.hidden_bn_layers = nn.ModuleList()
        for hli in range(hidden_layer_count):
            self.hidden_layers.append(nn.Linear(hidden_sample_count, hidden_sample_count))
            self.hidden_bn_layers.append(nn.BatchNorm1d(hidden_sample_count))

        self.fc_l2 = nn.Linear(hidden_sample_count, out_samples)
        self.bn2 = nn.BatchNorm1d(out_samples)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, input):
        if len(input.shape) == 3 and input.shape[1] == 1:
            input = input.view(-1, input.shape[2])
        elif len(input.shape) != 2:
            raise ArgumentError("dimensionality of data array is incorrect")

        x = F.leaky_relu(self.bn0(self.fc_l0(input)))

        for index, layer in enumerate(self.hidden_layers):
            bn_func = self.hidden_bn_layers[index]
            x = F.leaky_relu(bn_func(layer(x)))

        x = F.leaky_relu(self.bn2(self.fc_l2(x)))
        
        output = torch.tanh(x)
        return output

class LinearDiscriminator(nn.Module):
    def __init__(self, in_samples:int, out_samples:int, hidden_layer_count:int = 1, hidden_sample_count:int = 15):
        super(LinearDiscriminator, self).__init__()

        self.fc_l0 = nn.Linear(in_samples + out_samples, hidden_sample_count)
        self.bn0 = nn.BatchNorm1d(hidden_sample_count)

        hidden_layer_count = hidden_layer_count
        self.hidden_layers = nn.ModuleList()
        self.hidden_bn_layers = nn.ModuleList()
        for hli in range(hidden_layer_count):
            self.hidden_layers.append(nn.Linear(hidden_sample_count, hidden_sample_count))
            self.hidden_bn_layers.append(nn.BatchNorm1d(hidden_sample_count))

        self.fc_l2 = nn.Linear(hidden_sample_count, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, input, output):
        if len(input.shape) == 3 and input.shape[1] == 1:
            input = input.view(-1, input.shape[2])
        elif len(input.shape) != 2:
            raise ArgumentError("dimensionality of data array is incorrect")
        
        x = torch.cat([input, output], dim=1)

        x = F.leaky_relu(self.bn0(self.fc_l0(x)))

        for index, layer in enumerate(self.hidden_layers):
            bn_func = self.hidden_bn_layers[index]
            x = F.leaky_relu(bn_func(layer(x)))

        x = F.leaky_relu(self.fc_l2(x))
        
        output = torch.sigmoid(x)
        return output

data_set = CircleFullDataSet()

# Generator factory method used to allow gan object to create generator
# lazily (only when needed). can be exchanged easily
def generator_ctor():
    gNet = LinearGenerator(data_set.input_shape[1], data_set.output_shape[1])
    init_weights(gNet)
    return gNet

def discriminator_ctor():
    dNet = LinearDiscriminator(data_set.input_shape[1], data_set.output_shape[1])
    init_weights(dNet)
    return dNet

netG = generator_ctor()
netD = discriminator_ctor()

trainer = WassersteinGanNnTrainer(generator_neural_network=netG, discriminator_neural_network=netD, verbose=True, max_epoch_count=args["num_epochs"],batch_size=args["batch_size"], output_dir=OUTPUT_PATH, logger=LOGGER)


# Create callbacks for logging and plotting state
callbacks = [ 
    DataPlotterCallback(data_set.validation_set, args["sample_size"], netG, ScatterDataPlotter(), trainer.output_directory_path, logger=LOGGER),
]

# Gan instantiation and training here
trainer.train(data_set.training_set, data_set.training_set, callbacks=callbacks, metrics=[RMSE, MSE, MAE, AverageCosineSimilarity])