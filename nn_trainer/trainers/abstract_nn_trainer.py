from abc import abstractmethod
import io
import os
import os.path
from nn_trainer.logger import Logger

import json
from pathlib import Path
import shutil
import zipfile
import warnings
import copy
import numpy
from numpy import ndarray
from nn_trainer.models.networks.networks_1d import init_weights

import datetime

from nn_trainer.metrics import *
from nn_trainer.callbacks import *

import torch
import torch.nn
import torch.optim
import torch.utils.data as td
import torch.nn.functional as f

from typing import Dict

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy()
        elif isinstance(obj, torch.dtype):
            return str(obj)

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def make_path(output_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    return output_path

class AbstractNnTrainer(object):
    """
    Base class for Neural Network trainers
    """
    def __init__(
        self,
        neural_network: torch.nn.Module, 
        optimizer_fn: Any = torch.optim.Adam,
        optimizer_params: Dict = dict(lr=0.0001, betas=(0.5, 0.9)),
        scheduler_fn: Any = None,
        scheduler_params: Dict = field(default_factory=dict),
        dtype: Any = torch.float32,
        verbose: bool = False,
        patience: int = 0,
        max_epoch_count:int = 200,
        batch_size:int = 32,
        output_dir:str = os.path.join(os.getcwd(), "output"),
        logger:Logger = None,
        ):
        
        """
        The base constructor

        Args:
            neural_network (torch.nn.Module): 
                The network intended to be trained.  If implementation is a gan, then this refers to the 'generator' network
            optimizer_fn (Any, optional): 
                The factory method for the optimizer to be constructed and used for training during initialization. Defaults to torch.optim.Adam.
            optimizer_params (Dict, optional): 
                Parameters for the optimizer function. Defaults to dict(lr=2e-2).
            scheduler_fn (Any, optional): 
                The factory method for the optimizer scheduler to be constructed and used for training during initialization. Defaults to None.
            scheduler_params (Dict, optional): 
                The parameters for the optimizer scheduler function. Defaults to field(default_factory=dict).
            dtype (Any, optional): 
                The type of data to allocate memory with during training. Defaults to torch.float32.
            verbose (bool, optional): 
                Will print as much output as possible if True. Defaults to False.
            patience (int, optional): 
                Describes the stopping criteria number of epochs over which training losses are relatively the same. Defaults to 0.
            max_epoch_count (int, optional): 
                The maximum number of epochs to train for.
            batch_size (int, optional): 
                The number of samples in each batch.
            output_dir (str, optional): 
                The directory to store data pertinent to these models.
            logger (Logger, optional): 
                The logger to use when outputting information

        """
        self._verbose = verbose
        self._patience = patience
        self._logger = logger
        
        self._is_cuda_enabled = True if torch.cuda.is_available() else False
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._ngpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._dtype = dtype

        self._net_g = neural_network.type(dtype)
        self._net_g.to(self._device)
        self._net_g_optimizer_fn = optimizer_fn
        self._net_g_optimizer_params = optimizer_params
        self._net_g_scheduler_fn = scheduler_fn
        self._net_g_scheduler_params = scheduler_params

        self._net_g_optimizer = self._net_g_optimizer_fn(self._net_g.parameters(), **self._net_g_optimizer_params)

        self._epoch_count = max_epoch_count
        self._batch_size = batch_size
        self._model_directory_path = make_path(os.path.join(output_dir, "{}_{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), self.__class__.__name__))) 
        self._generator_model_directory_path = make_path(os.path.join(self._model_directory_path, self._net_g.__class__.__name__))
        self._most_recent_epoch_index_trained = 0

    @property
    def device(self):
        """
        The device on which memory is allocated and processing is performed during training

        Returns:
            torch.device: the torch device training is performed on, either 'cpu' or 'cuda:0'
        """
        return self._device

    @property
    def network(self):
        """
        The neural network indended to be trained

        Returns:
            [torch.nn.Module]: the neural network indended to be trained
        """
        return self._net_g

    @property
    def network_dtype(self):
        """
        The data type used for memory allocation in the network (mixed dtypes may be accommodated in the future)

        Returns:
            torch.dtype: the type of the data used for memory allocation (ex. torch.float32)
        """
        return self._dtype

    @property
    def output_directory_path(self):
        """
        The default output directory path for model persistence (to be removed in future instances in lieu of using appropriate callbacks)

        Returns:
            str: directory path of output data
        """
        return self._model_directory_path
    
    @property
    def most_recent_epoch_index_trained(self):
        """
        Gets the most recent epoch index this trainer trained model(s) on

        Returns:
            int: The epoch index last trained
        """
        return self._most_recent_epoch_index_trained

    def to_tensor(self, data:Any, dtype=torch.float32):
        """
        A utility method for converting data to tensor objects 

        Args:
            data (Any): The data to convert to a tensor
            dtype (torch.dtype, optional): Defaults to torch.float32.

        Raises:
            TypeError: If data is not a list, or not convertible to a numpy ndarray.

        Returns:
            torch.Tensor: A tensor
        """
        if isinstance(data, list):
            result = torch.tensor(numpy.asarray(data), dtype=dtype).to(device=self._device)
        elif isinstance(data, ndarray):
            result = torch.tensor(data, dtype=dtype).to(device=self._device)
        elif isinstance(data, torch.Tensor):
            result = data.clone().detach().requires_grad_(data.requires_grad).to(device=self._device, dtype=self._dtype)
        else:
            raise TypeError("type {} of data is not acceptable".format(type(data)))
        return result

    def to_numpy(self, data: torch.Tensor):
        """
        Converts a torch.Tensor to a numpy.ndarray

        Args:
            data (torch.Tensor): The array to convert

        Returns:
            numpy.ndarray: The converted array
        """
        if data.requires_grad:
            data = data.detach()
        if self._device == torch.device("cuda:0"):
            data = data.cpu()
        return data.numpy()

    @abstractmethod
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
            training_data_set (td.Dataset): 
                The dataset intended to be used for training
            validation_data_set (td.Dataset): 
                The dataset intended to be used for validation. all metrics defined will be invokd on this set
            callbacks (List[Callback], optional): 
                The set of callbacks receiving event based data. Defaults to [].
            loss_fn ([type], optional): 
                The loss function to be used as the objective function during training. Defaults to torch.nn.MSELoss().
            metrics (List[Metric], optional): 
                List of metrics to be invoked on validation data. Defaults to [].

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError("train method must be implemented")

    @abstractmethod
    def save_models(self) -> List[str]:
        """        
        Saves all models used by this trainer

        Raises:
            NotImplementedError: [description]

        Returns:
            List[str]: A list of paths for each model saved
        """
        raise NotImplementedError("train method must be implemented")

    def _save_model(self, directory:str, model_name:str):
        """Saving the model.  This method reuses the model output directory property 
        initialized during construction of this object.
        Parameters
        ----------
        model_name : str
            The name to give to the model. Recommendation: use epoch in the name
        Returns
        -------
        str
            input filepath with ".pt" appended
        """
        path = directory + f"\\{model_name}"

        saved_params = {}
        init_params = {}
        for key, val in self._net_g.state_dict().items():
            if isinstance(val, type):
                # Don't save torch specific params
                continue
            else:
                init_params[key] = { 'shape': val.shape, 'dtype': val.dtype }
        init_params["device_type"] = self._device.type
        init_params["device_index"] = self._device.index
        saved_params["init_params"] = init_params

        # Create folder
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save models params
        with open(Path(path).joinpath("model_params.json"), "w", encoding="utf8") as f:
            json.dump(saved_params, f, cls=ComplexEncoder, indent=4)

        # Save state_dict
        torch.save(self.network.state_dict(), Path(path).joinpath("network.pt"))
        shutil.make_archive(path, "zip", path)
        shutil.rmtree(path)
        print(f"Successfully saved model at {path}.zip")
        return f"{path}.zip"

    def load_model(self, filepath):
        """Load TabNet model.
        Parameters
        ----------
        filepath : str
            Path of the model.
        """
        path = Path(filepath)
        if path.suffix == "":
            path = Path(filepath + ".zip")
        
        if path.suffix != ".zip":
            raise TypeError("non zip files are not supported")

        try:
            with zipfile.ZipFile(path) as z:
                with z.open("model_params.json") as f:
                    loaded_params = json.load(f)
                    loaded_params["init_params"]["device_name"] = self._device
                with z.open("network.pt") as f:
                    try:
                        saved_state_dict = torch.load(f, map_location=self.device)
                    except io.UnsupportedOperation:
                        # In Python <3.7, the returned file object is not seekable (which at least
                        # some versions of PyTorch require) - so we'll try buffering it in to a
                        # BytesIO instead:
                        saved_state_dict = torch.load(io.BytesIO(f.read()), map_location=self.device)
        except KeyError:
            raise KeyError("Your zip file is missing at least one component")

        #self.__init__(**loaded_params["init_params"])

        #self._set_network()
        self.network.load_state_dict(saved_state_dict)
        self.network.eval()
        #self.load_class_attrs(loaded_params["class_attrs"])

        return

    @abstractmethod
    def _set_callbacks(self, custom_callbacks):
        raise NotImplementedError("_set_callbacks method must be implemented")
    
    def _set_metrics(self, metrics, eval_names):
        """Set attributes relative to the metrics.
        Parameters
        ----------
        metrics : list of str
            List of eval metric names.
        eval_names : list of str
            List of eval set names.
        """
        metrics = metrics or [RMSE]

        metrics = check_metrics(metrics)
        # Set metric container for each sets
        self._metric_container_dict = {}
        for name in eval_names:
            self._metric_container_dict.update({name: MetricContainer(metrics, prefix=f"{name}_metric__")})

        self._metrics = []
        self._metrics_names = []
        for _, metric_container in self._metric_container_dict.items():
            self._metrics.extend(metric_container.metrics)
            self._metrics_names.extend(metric_container.names)

        # Early stopping metric is the last eval metric
        self.early_stopping_metric = (self._metrics_names[-1] if len(self._metrics_names) > 0 else None)

    def _predict_epoch(self, name, data_loader):
        """
        Predict an epoch and update metrics.

        Parameters
        ----------
        name : str
            Name of the validation set
        loader : torch.utils.data.Dataloader
                DataLoader with validation set
        """
        # Setting network on evaluation mode
        self._net_g.eval()

        list_y_true = []
        list_y_score = []
        
        # Main loop
        for data in data_loader:
            X = data[0]
            y = data[1]
            scores = self._predict_batch(X)
            list_y_true.append(y)
            list_y_score.append(scores)

        y_true, scores = self._stack_batches(list_y_true, list_y_score)

        metrics_logs = self._metric_container_dict[name](y_true, scores)
        self._net_g.train()
        self.history.epoch_metrics.update(metrics_logs)
        return

    def _predict_batch(self, X):
        """
        Predict one batch of data.

        Parameters
        ----------
        X : torch.Tensor
            Owned products

        Returns
        -------
        np.array
            model scores
        """
        X = self.to_tensor(X)
        #X = X.to(self._device).float()

        # compute model output
        scores = self._net_g(X)

        if isinstance(scores, list):
            scores = [self.to_numpy(x) for x in scores]
        else:
            scores = self.to_numpy(scores)#cpu().detach().numpy()

        return scores
    
    def _stack_batches(self, list_y_true, list_y_score):
        y_true = np.vstack(list_y_true)
        y_score = np.vstack(list_y_score)
        return y_true, y_score