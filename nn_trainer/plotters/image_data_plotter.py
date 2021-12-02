import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn

from nn_trainer.plotters import DataPlotter
matplotlib.use('Agg')

def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path

class MnistCganImageDataPlotter(DataPlotter):
    def plot_batch(self, train_set_inputs, train_set_output, synthetic_output, epoch, output_dir):

        plt.clf()
        sample_dir = make_path(os.path.join(output_dir, str(epoch)))
        width = 0.4  # width of bar

        for idx, sample in enumerate(synthetic_output):
            output_path = os.path.join(sample_dir, "{}_noise".format(idx + 1))
            sample = sample[0]

            ax1 = plt.subplot(212)
            y_input = torch.nn.functional.one_hot(torch.tensor(train_set_inputs[idx,0]).long(), num_classes=10).numpy()
            x_input = np.arange(len(y_input))
            #y_input = train_set_inputs[idx,:]
            ax1.set_xticks(x_input)

            plt.bar(x_input, y_input, label='Input_{}'.format(0), color='indigo')
            plt.ylabel('Input {}'.format(0))
            plt.ylim(0, 1.1)
            plt.legend(loc=4)
            plt.grid(axis='x')

            ax2 = plt.subplot(221)
            plt.imshow(train_set_output[idx,0,:,:])
            plt.colorbar()

            ax3 = plt.subplot(222)
            plt.imshow(synthetic_output[idx,0,:,:])
            plt.colorbar()
            
            plt.savefig(output_path + '.png')
            plt.clf()
            plt.cla()
            plt.close()
        pass