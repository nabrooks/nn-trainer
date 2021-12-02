import os
import time
import math
import logging
import argparse
import numpy as np

from scipy import interpolate

# Training
EPOCHS = 150
BATCH_SIZE = 128

SAMPLE_EVERY = 10  # Generate audio samples every 1 epoch.
SAMPLE_NUM = 10  # Generate 10 samples every sample generation.

OUTPUT_PATH = os.path.join(os.getcwd(), "output")

def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

output = make_path(OUTPUT_PATH)

def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a 1d GAN with 2 input arrays on a given set 1d arrays')
    parser.add_argument('-lra', '--lrelu-alpha', dest='alpha', type=float, default=0.2, help='Slope of negative part of LReLU used by discriminator')
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=BATCH_SIZE, help='Batch size used for training')
    parser.add_argument('-ne', '--max-epoch-count', dest='max_epoch_count', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('-ld', '--latent-size', dest='latent_size', type=int, default=0, help='Size of latent dimension used by generator')
    parser.add_argument('-eps', '--epochs-per-sample', dest='epochs_per_sample', type=int, default=SAMPLE_EVERY, help='How many epochs between every set of samples generated for inspection')
    parser.add_argument('-ss', '--sample-size', dest='sample_size', type=int, default=SAMPLE_NUM, help='Number of inspection samples generated')
    parser.add_argument('-rf', '--regularization-factor', dest='lmbda', type=float, default=10.0, help='Gradient penalty regularization factor')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=0.0001, help='Initial ADAM learning rate')
    parser.add_argument('-bo', '--beta-one', dest='beta1', type=float, default=0.5, help='beta_1 ADAM parameter')
    parser.add_argument('-bt', '--beta-two', dest='beta2', type=float, default=0.9, help='beta_2 ADAM parameter')
    parser.add_argument('-v', '--verbose', dest='verbose', default=True, action='store_true')
    parser.add_argument('-od', '--output-dir', dest='output_dir', type=str, default=output, help='Path to directory where model files will be output')
    parser.add_argument('-tr', '--testing-ratio', dest='testing_ratio', type=int, default=1, help='The subet ratio of the data set to test on')
    parser.add_argument('-vr', '--validation-ratio', dest='validation_ratio', type=int, default=1, help='The subet ratio of the data set to validate on')
    parser.add_argument('-nsm', '--save-models-count', dest='save_models_count', type=int, default=15, help='The number of epoch based models to save from the last epoch')
    args = parser.parse_args()
    return vars(args)

def interpolate_1D(array, new_array_size):
    array_size = len(array)
    x = np.arange(0, array_size)
    y = array
    interpolation_function = interpolate.interp1d(x, y, fill_value="extrapolate")
    interpolation_ratio = array_size / new_array_size
    xnew = np.arange(0, array_size, interpolation_ratio)
    ynew = interpolation_function(xnew)
    return ynew

def interpolate_2D(array, new_array_shape):
    x_sz = array.shape[0]
    y_sz = array.shape[1]
    x = np.arange(0, x_sz)
    y = np.arange(0, y_sz)
    z = array
    interpolation_function = interpolate.interp2d(y, x, z)
    x_interpolation_ratio = x_sz / new_array_shape[0]
    y_interpolation_ratio = y_sz / new_array_shape[1]
    xnew = np.arange(0, x_sz, x_interpolation_ratio)
    ynew = np.arange(0, y_sz, y_interpolation_ratio)
    znew = interpolation_function(ynew, xnew)
    return znew
