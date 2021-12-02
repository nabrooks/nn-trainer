import os
import json
from nn_trainer.logger import Logger

import numpy
from nn_trainer.trainers import *
from nn_trainer.trainers.abstract_nn_trainer import AbstractNnTrainer

from nn_trainer.metrics import *
from nn_trainer.callbacks import *

import torch
import torch.nn
import torch.optim
import torch.utils.data as td
from torch.autograd import grad as torch_grad


from typing import Dict

def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path

class WassersteinGanNnTrainer(AbstractNnTrainer):
    """
    Basic implementation of a generative adversarial network trainer using Wasserstein loss metric
    and backpropagation

    Args:
        AbstractNnTrainer ([type]): The base class from which this class inherits
    """
    def __init__(
        self,
        generator_neural_network: torch.nn.Module, 
        discriminator_neural_network: torch.nn.Module, 
        generator_optimizer_fn: Any = torch.optim.Adam,
        generator_optimizer_params: Dict = dict(lr=0.0001, betas=(0.5, 0.9)),
        discriminator_optimizer_fn: Any = torch.optim.Adam,
        discriminator_optimizer_params: Dict = dict(lr=0.0001, betas=(0.5, 0.9)),
        scheduler_fn: Any = None,
        scheduler_params: Dict = field(default_factory=dict),
        dtype: Any = torch.float32,
        verbose: bool = False,
        patience: int = 0,
        critic_iterations: int = 5,
        gradient_penalty_weight: int = 10,
        max_epoch_count:int = 200,
        batch_size:int = 32,
        output_dir:str = os.path.join(os.getcwd(), "output"),
        logger:Logger = None
        ):
        """Constructor

        Args:
            generator_neural_network (torch.nn.Module): The generator network
            discriminator_neural_network (torch.nn.Module): The discriminator network
            generator_optimizer_fn (Any, optional): Defaults to torch.optim.Adam.
            generator_optimizer_params (Dict, optional): Defaults to dict(lr=0.0002).
            discriminator_optimizer_fn (Any, optional): Defaults to torch.optim.Adam.
            discriminator_optimizer_params (Dict, optional): Defaults to dict(lr=0.0002).
            scheduler_fn (Any, optional): Defaults to None.
            scheduler_params (Dict, optional): Defaults to field(default_factory=dict).
            dtype (Any, optional): Defaults to torch.float32.
            verbose (bool, optional): Defaults to False.
            patience (int, optional): Defaults to 0.
            max_epoch_count (int, optional): The maximum number of epochs to train for.
            batch_size (int, optional): The number of samples in each batch.
            output_dir (str, optional): The directory to store data pertinent to these models.
            logger (Logger, optional): Defaults to None.
        """
        super().__init__(
            neural_network=generator_neural_network, 
            optimizer_fn=generator_optimizer_fn,
            optimizer_params=generator_optimizer_params,
            scheduler_fn=scheduler_fn,
            scheduler_params=scheduler_params,
            dtype=dtype,
            verbose=verbose,
            patience=patience,
            max_epoch_count=max_epoch_count,
            batch_size=batch_size,
            output_dir=output_dir,
            logger=logger
            )
        self._critic_iterations = critic_iterations
        self._net_d = discriminator_neural_network.type(dtype)
        self._net_d.to(self._device)
        self._net_d_optimizer_fn = discriminator_optimizer_fn
        self._net_d_optimizer_params = discriminator_optimizer_params
        self._net_d_scheduler_fn = scheduler_fn
        self._net_d_scheduler_params = scheduler_params
        self._gradient_penalty_weight = gradient_penalty_weight
        self._net_d_optimizer = self._net_d_optimizer_fn(self._net_d.parameters(), **self._net_d_optimizer_params)

        self._discriminator_model_directory_path = make_path(os.path.join(self._model_directory_path, self._net_d.__class__.__name__))

    def train(
        self,
        training_data_set: td.Dataset,
        validation_data_set: td.Dataset, 
        callbacks: List[Callback] = [], 
        loss_fn = torch.nn.MSELoss(),
        metrics: List[Metric] = [],
        ):
        """        
        The training method

        Args:
            training_data_set (td.Dataset): The dataset intended to be used for training
            validation_data_set (td.Dataset): The dataset intended to be used for validation. all metrics defined will be invokd on this set
            callbacks (List[Callback], optional): The set of callbacks receiving event based data. Defaults to [].
            loss_fn ([type], optional): The loss function to be used as the objective function during training. Defaults to torch.nn.MSELoss().
            metrics (List[Metric], optional): List of metrics to be invoked on validation data. Defaults to [].

        Raises:
            NotImplementedError: [description]
        """
        self._set_metrics(metrics, ['validation'])
        self._set_callbacks(callbacks)
        self._loss_fn = loss_fn
        
        self._dataset_train = training_data_set
        self._dataset_valid = validation_data_set
        self._train_data_loader = td.DataLoader(dataset=self._dataset_train, batch_size=self._batch_size, shuffle=True)
        self._valid_data_loader = td.DataLoader(dataset=self._dataset_valid, batch_size=self._batch_size, shuffle=True)
        
        self._eval_names = ['validation']

        #with open(os.path.join(self._model_directory_path, 'training_config.json'), 'w') as f: json.dump(self._args, f, indent=4)

        self._stop_training = False
        
        
        num_steps = 0
        
        # Call method on_train_begin for all callbacks
        self._callback_container.on_train_begin()
        for epoch_index in range(self._epoch_count):
            self._most_recent_epoch_index_trained = epoch_index
            self._callback_container.on_epoch_begin(epoch_index)
            self._net_g.train()
            self._net_d.train()
            for i, (i_real, o_real) in enumerate(self._train_data_loader):
                num_steps += 1
                batch_logs = {"batch_size": i_real.shape[0]}

                i_real = self.to_tensor(i_real, self._dtype)
                o_real = self.to_tensor(o_real, self._dtype)
                
                # Get generated data
                o_fake = self.sample_generator(i_real)

                # Calculate probabilities on real and generated data
                #i_real = torch.autograd.Variable(i_real)
                #o_real = torch.autograd.Variable(o_real)
                
                d_real = self._net_d(i_real, o_real)
                d_fake = self._net_d(i_real, o_fake)

                # Get gradient penalty
                gradient_penalty, grad_norm = self._gradient_penalty(i_real, o_real, o_fake)
                #self.losses['GP'].append(gradient_penalty.data[0])

                # Create total loss and optimize
                self._net_d_optimizer.zero_grad()
                d_loss = d_fake.mean() - d_real.mean() + gradient_penalty
                d_loss.backward()

                self._net_d_optimizer.step()

                # Record loss
                #batch_logs["loss"] = self.to_numpy(d_loss).item()
                batch_logs["d_loss"] = self.to_numpy(d_loss).item()

                # Only update generator every |critic_iterations| iterations
                if num_steps % self._critic_iterations == 0:
                    """ """
                    self._net_g_optimizer.zero_grad()

                    # Get generated data
                    generated_data = self.sample_generator(i_real)

                    # Calculate loss and optimize
                    d_fake = self._net_d(i_real, generated_data)
                    g_loss = -d_fake.mean()
                    g_loss.backward()   
                    self._net_g_optimizer.step()

                    # Record loss
                    batch_logs['g_loss'] = self.to_numpy(g_loss).item()
                
                self._callback_container.on_batch_end(i, batch_logs)

                # if i % self.print_every == 0:
                #     print("Iteration {}".format(i + 1))
                #     print("D: {}".format(self.losses['D'][-1]))
                #     print("GP: {}".format(self.losses['GP'][-1]))
                #     print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                #     if self.num_steps > self.critic_iterations:
                #         print("G: {}".format(self.losses['G'][-1]))
            epoch_logs = { 
                "lr_net_g": self._net_g_optimizer.param_groups[-1]["lr"],
                "lr_net_d": self._net_d_optimizer.param_groups[-1]["lr"] 
            }
            self.history.epoch_metrics.update(epoch_logs)

            # Apply predict epoch to all validation set
            #for eval_name, valid_data in zip(self._eval_names, self._valid_data_loader):
            #    self._predict_epoch(eval_name, valid_data)

            for eval_name in self._eval_names:
                self._predict_epoch(eval_name, self._valid_data_loader)

            # Call method on_epoch_end for all callbacks
            self._callback_container.on_epoch_end(epoch_index, logs=self.history.epoch_metrics)
            if self._stop_training:
                break
        
        # Call method on_train_end for all callbacks
        self._callback_container.on_train_end()
        self._net_g.eval()
        self._net_d.eval()
    
    def save_models(self):
        """
        Saves all models associated with this trainer

        Returns:
            List[str]: List of paths associated with each model saved
        """
        netG_output_directory = self.output_directory_path + '\\Generator'
        netD_output_directory = self.output_directory_path + '\\Discriminator'
        make_path(netG_output_directory)
        make_path(netD_output_directory)
        modelG = self._save_model(netG_output_directory, f'epoch_{self.most_recent_epoch_index_trained}_{self._net_g.__class__.__name__}')
        modelD = self._save_model(netD_output_directory, f'epoch_{self.most_recent_epoch_index_trained}_{self._net_d.__class__.__name__}')
        return [ modelG, modelD ]

    def sample_generator(self, inputs):
        generated_data = self._net_g(inputs)
        return generated_data

    def _gradient_penalty(self, real_inputs, real_data, gen_data):
        batch_size = real_data.size()[0]
        dims = numpy.ones(len(real_data.shape), dtype=numpy.int64)
        dims[0] = batch_size
        dims = tuple(dims)
        t = torch.rand(dims, requires_grad=True)
        t = t.expand_as(real_data)

        if self._is_cuda_enabled:
            t = t.cuda()

        # mixed sample from real and fake; make approx of the 'true' gradient norm
        interpol = t * real_data.data + (1-t) * gen_data.data

        if self._is_cuda_enabled:
            interpol = interpol.cuda()

        #interpol = torch.tensor(interpol, requires_grad=True)
        
        prob_interpol = self._net_d(real_inputs, interpol)
        torch.autograd.set_detect_anomaly(True)
        gradients = torch_grad(outputs=prob_interpol, inputs=interpol, grad_outputs=torch.ones(prob_interpol.size()).cuda() if self._is_cuda_enabled else torch.ones(prob_interpol.size()), create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        #grad_norm = torch.norm(gradients, dim=1).mean()
        #self.losses['gradient_norm'].append(grad_norm.item())

        # add epsilon for stability
        eps = 1e-10
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1, dtype=torch.double) + eps)
        #gradients = gradients.cpu()
        # comment: precision is lower than grad_norm (think that is double) and gradients_norm is float
        return self._gradient_penalty_weight * (torch.max(torch.zeros(1,dtype=torch.double).cuda() if self._is_cuda_enabled else torch.zeros(1,dtype=torch.double), gradients_norm.mean() - 1) ** 2), gradients_norm.mean().item()

    def _set_callbacks(self, custom_callbacks):
        """Setup the callbacks functions.
        Parameters
        ----------
        custom_callbacks : list of func
            List of callback functions.
        """
        # Setup default callbacks history, early stopping and scheduler
        callbacks = []
        self.history = History(self, verbose=self._verbose, logger=self._logger)
        callbacks.append(self.history)
        if (self.early_stopping_metric is not None) and (self._patience > 0):
            early_stopping = EarlyStopping(early_stopping_metric=self.early_stopping_metric, is_maximize=(self._metrics[-1]._maximize if len(self._metrics) > 0 else None), patience=self._patience, logger=self._logger)
            callbacks.append(early_stopping)
        else:
            if self._logger is not None:
                self._logger.info("No early stopping will be performed, last training weights will be used.")
        if self._net_g_scheduler_fn is not None:
            # Add LR Scheduler call_back for generator
            is_batch_level = self._net_g_scheduler_params.pop("is_batch_level", False)
            net_g_scheduler = LRSchedulerCallback(scheduler_fn=self._net_g_scheduler_fn, scheduler_params=self._net_g_scheduler_params, optimizer=self._net_g_optimizer, early_stopping_metric=self.early_stopping_metric, is_batch_level=is_batch_level)
            callbacks.append(net_g_scheduler)
        if self._net_d_scheduler_fn is not None:
            # Add LR Scheduler call_back for discriminator
            is_batch_level = self._net_d_scheduler_params.pop("is_batch_level", False)
            net_d_scheduler = LRSchedulerCallback(scheduler_fn=self._net_d_scheduler_fn, scheduler_params=self._net_d_scheduler_params, optimizer=self._net_d_optimizer, early_stopping_metric=self.early_stopping_metric, is_batch_level=is_batch_level)
            callbacks.append(net_d_scheduler)

        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        self._callback_container = CallbackContainer(callbacks)
        self._callback_container.set_trainer(self)