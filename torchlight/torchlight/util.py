# in the name of God
#
import argparse
import os
import sys
import traceback
import time
import pickle
from collections import OrderedDict
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import logging


class IO():
    
    def __init__(self, work_dir, save_log=True, print_log=True):
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.cur_time = time.time()
        self.split_timer = {}
        self.model_text = ''
        
        if self.save_log:
            os.makedirs(self.work_dir, exist_ok=True)
        
        self.logger = logging.getLogger('IOLogger')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m.%d.%y|%X')
        
        if self.print_to_screen:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        if self.save_log:
            file_handler = logging.FileHandler(os.path.join(self.work_dir, 'log.txt'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    
    def log(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        self.logger.info(message)
    
    
    def load_model(self, model, **model_args):
        """
        Dynamically imports and initializes a model.
        """
        Model = import_class(model)
        model = Model(**model_args)
        self.model_text += '\n\n' + str(model)
        return model
    
    
    def load_weights(self, model, weights_path, ignore_weights=None, fix_weights=False):
        """
        Loads weights into the model, with options to ignore certain weights and fix them.
        """
        if ignore_weights is None:
            ignore_weights = []
        if isinstance(ignore_weights, str):
            ignore_weights = [ignore_weights]
        
        self.print_log(f'Load weights from {weights_path}.')
        weights = torch.load(weights_path)
        weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in weights.items()])
        
        for i in ignore_weights:
            ignore_name = [w for w in weights if w.startswith(i)]
            for n in ignore_name:
                weights.pop(n)
                self.print_log(f'Filter [{i}] remove weights [{n}].')
        
        for w in weights:
            self.print_log(f'Load weights [{w}].')
        
        try:
            model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = model.state_dict()
            diff = set(state.keys()).difference(set(weights.keys()))
            for d in diff:
                self.print_log(f'Cannot find weights [{d}].')
            state.update(weights)
            model.load_state_dict(state)
        
        if fix_weights:
            for name, param in model.named_parameters():
                if name in weights:
                    param.requires_grad = False
                    self.print_log(f'Fix weights [{name}].')
        
        return model
    
    
    def save_pkl(self, result, filename):
        """
        Saves a Python object to a pickle file.
        """
        with open(os.path.join(self.work_dir, filename), 'wb') as f:
            pickle.dump(result, f)
    
    
    def save_h5(self, result, filename, append=False):
        """
        Saves a dictionary of results to an HDF5 file.
        """
        mode = 'a' if append else 'w'
        with h5py.File(os.path.join(self.work_dir, filename), mode) as f:
            for k, v in result.items():
                f[k] = v
    
    
    def save_model(self, model, name):
        """
        Saves the model's state dictionary to a file.
        """
        model_path = os.path.join(self.work_dir, name)
        state_dict = model.state_dict()
        weights = OrderedDict([[k.replace('module.', ''), v.cpu()] for k, v in state_dict.items()])
        torch.save(weights, model_path)
        self.print_log(f'The model has been saved as {model_path}.')
    
    
    def save_arg(self, arg):
        """
        Saves command-line arguments to a YAML configuration file.
        """
        arg_dict = vars(arg)
        os.makedirs(self.work_dir, exist_ok=True)
        config_path = os.path.join(self.work_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f, default_flow_style=False, indent=4)
    
    
    def print_log(self, message, print_time=True):
        """
        Prints a log message to both the console and the log file.
        """
        if print_time:
            message = time.strftime("[%m.%d.%y|%X] ", time.localtime()) + message
        self.logger.info(message)
    
    
    def init_timer(self, *names):
        """
        Initializes timers for tracking different parts of the code.
        """
        self.record_time()
        self.split_timer = {name: 0.0000001 for name in names}
    
    
    def check_time(self, name):
        """
        Updates the timer for a specific section.
        """
        self.split_timer[name] += self.split_time()
    
    
    def record_time(self):
        """
        Records the current time.
        """
        self.cur_time = time.time()
        return self.cur_time
    
    
    def split_time(self):
        """
        Calculates the elapsed time since the last recorded time.
        """
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    
    
    def print_timer(self):
        """
        Prints the time consumption for each tracked section.
        """
        total = sum(self.split_timer.values())
        proportion = {
            name: f'{int(round((time_spent / total) * 100)):02d}%'
            for name, time_spent in self.split_timer.items()
        }
        self.print_log('Time consumption:')
        for name, pct in proportion.items():
            self.print_log(f'\t[{name}][{pct}]: {self.split_timer[name]:.4f}')


def str2bool(v):
    """
    Converts a string to a boolean.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2dict(v):
    """
    Converts a string representation of a dictionary to an actual dictionary.
    """
    return eval(f'dict({v})')  # pylint: disable=W0123


def _import_class_0(name):
    """
    Imports a class from a string name.
    """
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def import_class(import_str):
    """
    Imports a class from a string, handling errors if the class cannot be found.
    """
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(f'Class {class_str} cannot be found ({traceback.format_exc()})')


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  # pylint: disable=W0123
        output_dict = getattr(namespace, self.dest, {})
        output_dict.update(input_dict)
        setattr(namespace, self.dest, output_dict)

#cloner174