import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from nn_trainer.plotters import DataPlotter
matplotlib.use('Agg')

def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path

class LineChartDataPlotter(DataPlotter):
    def plot_batch(self, train_set_inputs, train_set_output, synthetic_output, epoch, output_dir):
        """
        Save output samples.
        """
        if len(train_set_inputs.shape) == 2 and len(train_set_output.shape) == 2 and len(synthetic_output.shape) == 2:
            train_set_inputs = train_set_inputs.reshape(train_set_inputs.shape[0], 1, train_set_inputs.shape[1])
            train_set_output = train_set_output.reshape(train_set_output.shape[0], 1, train_set_output.shape[1])
            synthetic_output = synthetic_output.reshape(synthetic_output.shape[0], 1, synthetic_output.shape[1])

        assert synthetic_output.shape == train_set_output.shape
        assert len(train_set_inputs.shape) == 3 and len(train_set_output.shape) == 3 and len(synthetic_output.shape) == 3

        plt.clf()
        sample_dir = make_path(os.path.join(output_dir, str(epoch)))
        width = 0.4  # width of bar

        for idx, sample in enumerate(synthetic_output):
            output_path = os.path.join(sample_dir, "{}_noise".format(idx + 1))

            sample = sample[0]
            x_output = np.arange(len(sample))
            nrows = train_set_inputs.shape[1] + train_set_output.shape[1]
            plt.figure(figsize=(20, 10))
            for ti in range(train_set_inputs.shape[1]):
                train_input = train_set_inputs[idx][ti]
                x_input = np.arange(len(train_input))
                ax = plt.subplot(nrows, 1, ti + 1)
                ax.set_ylim([-1, 1])
                #ax.set_xticks(x_input)

                plt.plot(x_input, train_input, label='Input_{}'.format(ti), color='indigo')
                plt.ylabel('Input {}'.format(ti))

                plt.ylim(-1.1, 1.1)
                plt.legend(loc=4)
                plt.grid(axis='x')

            train_set_sample = train_set_output[idx][0]
            # share x and y
            ax3 = plt.subplot(nrows, 1, nrows)
            ax3.set_ylim([-1, 1])
            #ax3.set_xticks(x_output)

            plt.plot(x_output, sample, label='G_output', color='dodgerblue')
            plt.plot(x_output, train_set_sample, label='Validation_Output', color='mediumvioletred')

            plt.xlabel('Sample Index')
            plt.ylabel('G Output')

            plt.legend(loc=4)
            #plt.grid(axis='x')
            plt.tight_layout()

            plt.savefig(output_path + '.png')
            plt.clf()
            plt.close()

