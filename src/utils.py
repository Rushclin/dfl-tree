import os
import sys
import torch
import random
import logging
import subprocess
import numpy as np

from tqdm import tqdm
from importlib import import_module
from collections import defaultdict
from multiprocessing import Process

logger = logging.getLogger(__name__)

class Range:
    """
    Class to define a range object with start and end values.
    It provides a custom comparison method and a string representation.
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __eq__(self, other):
        return self.start <= other <= self.end
    
    def __str__(self):
        return f'Specified Range: [{self.start:.2f}, {self.end:.2f}]'


def set_seed(seed: int = 32):
    """
    Seed all random number generators to ensure reproducibility.
    
    This function sets the seed for all key libraries and system environments that use random 
    number generation to ensure consistent results across multiple runs. It also configures 
    PyTorch to ensure deterministic behavior in operations that could otherwise be non-deterministic.
    
    Args:
        seed (int): The seed value to use for the random number generators. Default is 32.
    """
    torch.manual_seed(seed)    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f'[SEED] : {seed}!')

    
class TqdmToLogger(tqdm):
    """
    Custom progress bar class that integrates TQDM with the logger for smooth logging of progress.
    """
    def __init__(self, *args, logger=None, 
                 mininterval=0.1, 
                 bar_format='{desc:<}{percentage:3.0f}% |{bar:20}| [{n_fmt:6s}/{total_fmt}]', 
                 desc=None, 
                 **kwargs):
        self._logger = logger  # Custom logger can be passed
        super().__init__(*args, mininterval=mininterval, bar_format=bar_format, ascii=True, desc=desc, **kwargs)

    @property
    def logger(self):
        """
        Return the logger to be used, or fallback to the default logger.
        """
        if self._logger is not None:
            return self._logger
        return logger

    def display(self, msg=None, pos=None):
        """
        Display the progress bar message using the logger instead of the console.
        """
        if not self.n:
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg.strip('\r\n\t '))

def init_weights(model, init_type, init_gain):
    """
    Initialize the model weights based on the given initialization type and gain.
    
    Args:
        model: PyTorch model to initialize.
        init_type: The type of initialization to apply ('normal', 'xavier', etc.).
        init_gain: The gain (scaling factor) for the weight initialization.
    """
    def init_func(m): 
        classname = m.__class__.__name__
        # Handle BatchNorm initialization separately
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight.data, mean=1.0, std=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        # Initialize weights for Linear and Conv layers
        elif hasattr(m, 'weight') and (classname.find('Linear') == 0 or classname.find('Conv') == 0):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, mean=0., std=init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'truncnorm':
                torch.nn.init.trunc_normal_(m.weight.data, mean=0., std=init_gain)
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'none':  # PyTorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_func)

def check_args(args):
    """
    Validate and check the command-line arguments for potential issues, ensuring proper configurations.
    
    Args:
        args: Command-line arguments passed to the program.
        
    Returns:
        args: Validated and updated arguments with derived settings.
    """
    if 'cuda' in args.device:
        assert torch.cuda.is_available(), 'GPU not found'

    if args.optimizer not in torch.optim.__dict__.keys():
        err = f'`{args.optimizer}` Optimizer not found'
        logger.exception(err)
        raise AssertionError(err)
    
    if args.criterion not in torch.nn.__dict__.keys():
        err = f'`{args.criterion}` Loss function not found'
        logger.exception(err)
        raise AssertionError(err)

    
class MetricManager:
    """
    MetricManager is responsible for tracking and aggregating evaluation metrics during training/testing.
    """
    def __init__(self, eval_metrics):
        """
        Initialize the MetricManager with the evaluation metrics provided.
        
        Args:
            eval_metrics: A list of metrics to evaluate during training/testing.
        """
        self.metric_funcs = {
            name: import_module(f'.metrics', package=__package__).__dict__[name.title()]()
            for name in eval_metrics
        }
        self.figures = defaultdict(int)  # Store running figures for metrics
        self._results = dict()

        # If using Youden's J statistic in any metric, propagate that option
        if 'youdenj' in self.metric_funcs:
            for func in self.metric_funcs.values():
                if hasattr(func, '_use_youdenj'):
                    setattr(func, '_use_youdenj', True)

    def track(self, loss, pred, true):
        """
        Track the loss and update the metrics with new predictions and true values.
        
        Args:
            loss: The loss value for the current batch.
            pred: The predictions made by the model.
            true: The ground truth values.
        """
        self.figures['loss'] += loss * len(pred)

        for module in self.metric_funcs.values():
            module.collect(pred, true)

    def aggregate(self, total_len, curr_step=None):
        """
        Aggregate the collected metrics over the total dataset and calculate final results.
        
        Args:
            total_len: The total number of samples processed.
            curr_step: Optional argument to log results by training step.
        """
        running_figures = {name: module.summarize() for name, module in self.metric_funcs.items()}
        running_figures['loss'] = self.figures['loss'] / total_len
        if curr_step is not None:
            self._results[curr_step] = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
            }
        else:
            self._results = {
                'loss': running_figures['loss'], 
                'metrics': {name: running_figures[name] for name in self.metric_funcs.keys()}
            }
        self.figures = defaultdict(int)  # Reset for next aggregation

    @property
    def results(self):
        """
        Return the aggregated results after training/testing.
        
        Returns:
            dict: A dictionary containing loss and metrics results.
        """
        return self._results
 