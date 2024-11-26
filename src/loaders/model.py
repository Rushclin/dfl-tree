import torch
import logging
import inspect
import importlib

logger = logging.getLogger(__name__)

def load_model(args):
    model_class = importlib.import_module('..models', package=__package__).__dict__[args.model_name]

    required_args = inspect.getfullargspec(model_class)[0]

    model_args = {}
    for argument in required_args:
        if argument == 'self': 
            continue
        model_args[argument] = getattr(args, argument)

    model = model_class(**model_args)
    
    logger.info(f"[LOAD] : {args.model_name}")
    
    return model

def load_experiment_model(args):
    model = load_model(args)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model.to(args.device)
    return model
    