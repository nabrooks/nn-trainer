import os
import time
import datetime
import copy
import random

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any
import torch
import torch.nn
import torch.utils.data
import matplotlib.pyplot as plt

from nn_trainer.plotters import DataPlotter

class Callback:
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, model):
        self.trainer = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

@dataclass
class CallbackContainer:
    """
    Container holding a list of callbacks.
    """

    callbacks: List[Callback] = field(default_factory=list)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs["start_time"] = time.time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

@dataclass
class EarlyStopping(Callback):
    """EarlyStopping callback to exit the training loop if early_stopping_metric
    does not improve by a certain amount for a certain
    number of epochs.
    Parameters
    ---------
    early_stopping_metric : str
        Early stopping metric name
    is_maximize : bool
        Whether to maximize or not early_stopping_metric
    tol : float
        minimum change in monitored value to qualify as improvement.
        This number should be positive.
    patience : integer
        number of epochs to wait for improvement before terminating.
        the counter be reset after each improvement
    """

    early_stopping_metric: str
    is_maximize: bool
    tol: float = 0.0
    patience: int = 5
    logger: Any = None    

    def __post_init__(self):
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0
        self.best_weights = None
        self.best_loss = np.inf
        if self.is_maximize:
            self.best_loss = -self.best_loss
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.early_stopping_metric)
        if current_loss is None:
            return

        loss_change = current_loss - self.best_loss
        max_improved = self.is_maximize and loss_change > self.tol
        min_improved = (not self.is_maximize) and (-loss_change > self.tol)
        if max_improved or min_improved:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.wait = 1
            self.best_weights = copy.deepcopy(self.trainer.network.state_dict())
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.trainer._stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        self.trainer.best_epoch = self.best_epoch
        self.trainer.best_cost = self.best_loss

        if self.best_weights is not None:
            self.trainer.network.load_state_dict(self.best_weights)

        if self.stopped_epoch > 0:
            msg = f"\nEarly stopping occurred at epoch {self.stopped_epoch}"
            msg += (
                f" with best_epoch = {self.best_epoch} and "
                + f"best_{self.early_stopping_metric} = {round(self.best_loss, 5)}"
            )
            if self.logger is not None:
                self.logger.info(msg)
        else:
            msg = (
                f"Stop training because you reached max_epochs = {self.trainer.max_epochs}"
                + f" with best_epoch = {self.best_epoch} and "
                + f"best_{self.early_stopping_metric} = {round(self.best_loss, 5)}"
            )
            if self.logger is not None:
                self.logger.info(msg)
        if self.logger is not None:
            self.logger.info("Best weights from best epoch are automatically used!")

@dataclass
class History(Callback):
    """Callback that records events into a `History` object.
    This callback is automatically applied to
    every SuperModule.
    Parameters
    ---------
    trainer : DeepRecoModel
        Model class to train
    verbose : int
        Print results every verbose iteration
    """

    trainer: Any
    verbose: int = 1
    logger: Any = None

    def __post_init__(self):
        super().__init__()
        self.samples_seen = 0.0
        self.total_time = 0.0

    def on_train_begin(self, logs=None):
        self.history = {}#{"loss": []}
        self.history.update({"lr": []})
        self.history.update({name: [] for name in self.trainer._metrics_names})
        self.start_time = logs["start_time"]
        self.epoch_loss = 0.0
        self.batch_logs = {}
        self.previous_batch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_metrics = {}# {"loss": 0.0}
        self.samples_seen = 0.0

    def on_epoch_end(self, epoch, logs=None):
        #self.epoch_metrics["loss"] = self.epoch_loss
        for metric_name, metric_value in self.epoch_metrics.items():
            if metric_name not in self.history.keys():
                self.history[metric_name] = []
            self.history[metric_name].append(metric_value)
        for batch_log_name, batch_log_value in self.batch_logs.items():
            if batch_log_name not in self.history.keys():
                self.history[batch_log_name] = []
            self.history[batch_log_name].append(sum(batch_log_value) / len(batch_log_value))
        if self.verbose == 0:
            return
        if epoch % self.verbose != 0:
            return
        msg = f"epoch {epoch:<3}"
        for metric_name, metric_value in self.epoch_metrics.items():
            if metric_name != "lr":
                msg += f"| {metric_name:<3}: {np.round(metric_value, 5):<8}"
        self.total_time = int(time.time() - self.start_time)
        msg += f"|  {str(datetime.timedelta(seconds=self.total_time)) + 's':<6}"
        if self.logger is not None:
            self.logger.info(msg)

    def on_batch_end(self, batch, logs=None):
        batch_size = logs["batch_size"]
        
        # if epoch just started, clear batch logs
        if self.previous_batch > batch:
            self.batch_logs.clear()
        
        for key in logs:
            if key not in self.batch_logs:
                self.batch_logs[key] = list()
            self.batch_logs[key].append(logs[key])
        self.samples_seen += batch_size
        batch_count = int(len(self.trainer._dataset_train) / batch_size)
        self.previous_batch = batch
        if self.logger is not None:
            self.logger.info("Batch {} of {} batches trained. Batch size: {}".format(batch, batch_count, batch_size))

    def __getitem__(self, name):
        return self.history[name]

    def __repr__(self):
        return str(self.history)

    def __str__(self):
        return str(self.history)

@dataclass
class LRSchedulerCallback(Callback):
    """Wrapper for most torch scheduler functions.
    Parameters
    ---------
    scheduler_fn : torch.optim.lr_scheduler
        Torch scheduling class
    scheduler_params : dict
        Dictionnary containing all parameters for the scheduler_fn
    is_batch_level : bool (default = False)
        If set to False : lr updates will happen at every epoch
        If set to True : lr updates happen at every batch
        Set this to True for OneCycleLR for example
    """

    scheduler_fn: Any
    optimizer: Any
    scheduler_params: dict
    early_stopping_metric: str
    is_batch_level: bool = False

    def __post_init__(
        self,
    ):
        self.is_metric_related = hasattr(self.scheduler_fn, "is_better")
        self.scheduler = self.scheduler_fn(self.optimizer, **self.scheduler_params)
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        if self.is_batch_level:
            self.scheduler.step()
        else:
            pass

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.early_stopping_metric)
        if current_loss is None:
            return
        if self.is_batch_level:
            pass
        else:
            if self.is_metric_related:
                self.scheduler.step(current_loss)
            else:
                self.scheduler.step()
@dataclass
class TrainerSaverCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        return
        

@dataclass
class DataPlotterCallback(Callback):
    data_set: torch.utils.data.Dataset
    data_set_sample_count: int
    network_trainer: Any
    data_plotter: DataPlotter
    shuffle: bool = True
    logger: Any = None

    def __post_init__(self):
        super().__init__()
        self.epoch_logs = dict()
        self.epoch_logs_mins = dict()
        self.epoch_logs_maxs = dict()
        self.batch_logs = dict()
        self.batch_logs_min = 0
        self.batch_logs_max = 0
        self.inputs_outputs_real = []
        self.previous_batch = 0
        
        sample_indices = list(random.sample(range(0, len(self.data_set)-1), self.data_set_sample_count)) if self.shuffle else list(range(0, self.data_set_sample_count))
        
        for sample_index in sample_indices:
            input_real, output_real = self.data_set.__getitem__(sample_index)
            self.inputs_outputs_real.append((input_real, output_real))
    
    def on_batch_end(self, batch, logs=None):
        if self.previous_batch > batch:
            self.batch_logs.clear()

        for key in logs:
            if key not in self.batch_logs.keys():
                self.batch_logs[key] = list()
            self.batch_logs[key].append(logs[key])

        self.previous_batch = batch
        return

    def on_epoch_end(self, epoch, logs=None):
        inputs_real = []
        outputs_real = []
        outputs_fake = []
                
        dtype = self.trainer.network_dtype
        

        for input_real, output_real in self.inputs_outputs_real:
            inputs_real.append(input_real)
            outputs_real.append(output_real)

            input_real = self.trainer.to_tensor([input_real], dtype=dtype)
            
            self.trainer.network.eval()
            with torch.set_grad_enabled(False):
                output_fake = self.trainer.network(input_real)   
            self.trainer.network.train()

            output_fake = output_fake[0]
            output_fake = self.trainer.to_numpy(output_fake)
            
            outputs_fake.append(output_fake)
        

        inputs_real = np.asarray(inputs_real)
        outputs_real = np.asarray(outputs_real)
        outputs_fake = np.asarray(outputs_fake)

        for key in logs:
            if key not in self.epoch_logs.keys():
                self.epoch_logs[key] = list()
            self.epoch_logs[key].append(logs[key])
        
        for key in self.batch_logs:
            if key not in self.epoch_logs.keys():
                self.epoch_logs[key] = list()
                self.epoch_logs_mins[key] = list()
                self.epoch_logs_maxs[key] = list()
            self.epoch_logs[key].append(sum(self.batch_logs[key])/len(self.batch_logs[key]))
            self.epoch_logs_mins[key].append(min(self.batch_logs[key]))
            self.epoch_logs_maxs[key].append(max(self.batch_logs[key]))

        # plot learning rate logs on same plot
        learning_rate_keys = [x for x in logs.keys() if 'lr_' in x or 'lr' in x]
        lr_logs = {}
        for lr_key in learning_rate_keys:
            lr_logs[lr_key] = self.epoch_logs[lr_key]
        self._plot_epoch_logs('learning_rates', lr_logs, self.trainer.output_directory_path, fill_to_zero=False)

        # plot network loss logs on same plot
        loss_keys = [x for x in self.batch_logs.keys() if 'loss' in x]
        self._plot_epoch_logs_windowed('losses', loss_keys, self.trainer.output_directory_path)

        # plot metric logs separately
        metric_keys = [x for x in logs.keys() if 'loss' not in x and 'lr_' not in x or 'lr' not in x]
        metric_logs = {}
        for metric_key in metric_keys:
            metric_logs[metric_key] = self.epoch_logs[metric_key]
            self._plot_epoch_log(metric_key, metric_logs[metric_key], self.trainer.output_directory_path)

        # plot qc images
        if self.logger is not None:
            self.logger.info("Plotting {} test samples".format(inputs_real.shape[0]))

        #utils.LOGGER.info("Plotting {} test samples".format(inputs_real.shape[0]))
        self.data_plotter.plot_batch(inputs_real, outputs_real, outputs_fake, epoch, self.trainer.output_directory_path)
    
    def _plot_epoch_logs_windowed(self, title, epoch_dictionary_keys, output_directory_path):
        if len(epoch_dictionary_keys) == 0 or epoch_dictionary_keys is None:
            return 
        
        for key in epoch_dictionary_keys:
            escape = False
            if key not in self.epoch_logs.keys():
                var_name = [ i for i, a in locals().iteritems() if a == self.epoch_logs][0]
                if self.logger is not None:
                    self.logger.info("Cannot plot filled plot because key: {} is not in dictionary: {}".format(key, var_name))
                escape = True
            if key not in self.epoch_logs_mins.keys():
                var_name = [ i for i, a in locals().iteritems() if a == self.epoch_logs][0]
                if self.logger is not None:
                    self.logger.info("Cannot plot filled plot because key: {} is not in dictionary: {}".format(key, var_name))
                escape = True
            if key not in self.epoch_logs_maxs.keys():
                var_name = [ i for i, a in locals().iteritems() if a == self.epoch_logs][0]
                if self.logger is not None:
                    self.logger.info("Cannot plot filled plot because key: {} is not in dictionary: {}".format(key, var_name))
                #utils.LOGGER.info("Cannot plot filled plot because key: {} is not in dictionary: {}".format(key, var_name))
                escape = True
            if escape == True:
                return

        first = self.epoch_logs[epoch_dictionary_keys[0]]
        epoch_x = np.arange(0, len(first))

        for key in epoch_dictionary_keys:
            epoch_y = self.epoch_logs[key]
            epoch_y_mins = self.epoch_logs_mins[key]
            epoch_y_maxs = self.epoch_logs_maxs[key]
            
            plt.plot(epoch_x, epoch_y, label=key)
            plt.fill_between(epoch_x, epoch_y_mins, epoch_y_maxs, alpha=0.2)
            plt.annotate('%0.5f' % epoch_y[len(epoch_y) - 1], xy=(1, epoch_y[len(epoch_y) - 1]), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')

            plt.legend(loc=4)
            plt.grid(True)
            plt.tight_layout()
            plt.xlabel('Epoch')
            #plt.yscale('log')

        plt.savefig(os.path.join(output_directory_path, "{}.png".format(title)))
        plt.clf()
        plt.cla()
        plt.close()
        
        return 

    def _plot_epoch_logs(self, title, data_dictionary, output_directory_path, fill_to_zero:bool = True):
        first = data_dictionary[list(data_dictionary.keys())[0]]
        x = np.arange(0, len(first))

        for key in data_dictionary.keys():
            y = data_dictionary[key]
            plt.plot(x, y, label=key)

            if fill_to_zero == True:
                plt.fill_between(x, np.zeros(len(x)), y, alpha=0.2)

            plt.annotate('%0.5f' % y[len(y) - 1], xy=(1, y[len(y) - 1]), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')

            plt.legend(loc=4)
            plt.grid(True)
            plt.tight_layout()
            plt.xlabel('Epoch')

        plt.savefig(os.path.join(output_directory_path, "{}.png".format(title)))
        plt.clf()
        plt.cla()
        plt.close()
    
    def _plot_epoch_log(self, title, data, output_directory_path, fill_to_zero: bool=True):
        x = np.asarray(range(0, len(data)))
        y = data

        plt.figure(figsize=(20, 10))
        plt.plot(x, y, label=title, color='darkorange')

        if fill_to_zero == True:
            plt.fill_between(x, np.zeros(len(x)), y, alpha=0.2)
        #plt.ylim(-0.1, 2.1)
        plt.annotate('%0.5f' % y[len(y) - 1], xy=(1, y[len(y) - 1]), xytext=(8, 0), xycoords=('axes fraction', 'data'), textcoords='offset points')

        plt.legend(loc=4)
        plt.grid(True)
        plt.tight_layout()
        plt.xlabel('Epoch')

        plt.savefig(os.path.join(output_directory_path, "{}.png".format(title)))
        plt.clf()
        plt.cla()
        plt.close()