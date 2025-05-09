# -*- coding: utf-8 -*-

import torch
import logging
import sys
import pdb
from typing import Dict, List, Tuple, Iterable, Callable
from torch.nn import Module, ModuleList
import copy, os
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math

class ChildnetAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in
    `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`__.
    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-7,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        reserve_p = 1.0,
        tuning_mode = 'ad_tuning'
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

        self.grad_mask = None
        self.reserve_p = reserve_p
        self.tuning_mode = tuning_mode
        print(f"lr = {lr}, reserve_p = {reserve_p}, tuning_mode = {tuning_mode}")

    def set_grad_mask(self, grad_mask):
        self.grad_mask = grad_mask

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # =================== HACK BEGIN =======================         
                if self.tuning_mode is not None:
                    if self.tuning_mode == 'ad_tuning':
                        if p in self.grad_mask:
                            grad *= self.grad_mask[p]
                    else:
                        # grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                        # grad *= grad_mask.sample() 
                        raise("not implement !!!")
                else: 
                    raise("not implement !!!")
                # =================== HACK END =======================

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
        
    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if len(state) == 0:
                    pass
                else:
                    lr.append(group['lr'])
        return lr

def save_checkpoint(model, optimizer, filename):
    try:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'configs': model.configs}, filename)
    except:
        torch.save({'model': model.state_dict(), \
            'optimizer_tsvad': optimizer['tsvad'].state_dict(), \
            'optimizer_resnet': optimizer['resnet'].state_dict(), 'configs': model.configs},  filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        
def load_checkpoint_join_training(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    # pdb.set_trace()
    if model is not None:
        model_dict = model.state_dict()
        # pdb.set_trace()
        state_dict_2 = {k:v for k,v in checkpoint['model'].items()}
        # pdb.set_trace()
        print(state_dict_2.keys())
        # for k,v in state_dict_2:
            # print(k)
        model_dict.update(state_dict_2)
        model.load_state_dict(model_dict)
        # model_dict['FC.2.weight'] - checkpoint['model']['FC.2.weight']
        # pdb.set_trace()
        # model.load_state_dict(checkpoint['model'])
    # pdb.set_trace()
    if optimizer is not None and 'join_train' in filename:
        print('load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])

def load_checkpoint_join_training_init(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    # pdb.set_trace()
    if model is not None:
        model_dict = model.state_dict()
        # pdb.set_trace()
        state_dict_2 = {k.replace('module', 'css_model'):v for k,v in checkpoint['model'].items() if k.replace('module', 'css_model') in model_dict and model_dict[k].size() == v.size()}
        update_key = {k for k in state_dict_2.keys() if k in model_dict.keys()}
        print(update_key)
        # pdb.set_trace()
        # print(state_dict_2.keys())
        # for k in state_dict_2.keys():
            # print(k)
            # pdb.set_trace()
        model_dict.update(state_dict_2)
        model.load_state_dict(model_dict)
        # model_dict['FC.2.weight'] - checkpoint['model']['FC.2.weight']
        # pdb.set_trace()
        # model.load_state_dict(checkpoint['model'])
    # pdb.set_trace()
    if optimizer is not None and 'join_train' in filename:
        print('load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])

def load_checkpoint_join_training_init_mask(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    # pdb.set_trace()
    if model is not None:
        model_dict = model.state_dict()
        # pdb.set_trace()
        state_dict_2 = {k.replace('module', 'css_model'):v for k,v in checkpoint['model'].items() \
                     if k.replace('module', 'css_model') in model_dict \
                     and model_dict[k.replace('module', 'css_model')].size() == v.size()}
        not_update_key = {k for k in model_dict.keys() if k not in state_dict_2.keys()}
        update_key = {k for k in state_dict_2.keys() if k in model_dict.keys()}
        print("not_update_key: ", not_update_key)
        print("update_key: ", update_key)
        # pdb.set_trace()
        # print(state_dict_2.keys())
        # for k in state_dict_2.keys():
            # print(k)
            # pdb.set_trace()
        model_dict.update(state_dict_2)
        model.load_state_dict(model_dict)
        # model_dict['FC.2.weight'] - checkpoint['model']['FC.2.weight']
        # pdb.set_trace()
        # model.load_state_dict(checkpoint['model'])
    # pdb.set_trace()
    if optimizer is not None and 'join_train' in filename:
        print('load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        
def load_checkpoint_join_training_init_mask_decode(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    # pdb.set_trace()
    if model is not None:
        model_dict = model.state_dict()
        # pdb.set_trace()
        state_dict_2 = {k.replace('module', 'css_model'):v for k,v in checkpoint['model'].items() \
                     if k.replace('module', 'css_model') in model_dict \
                     and model_dict[k.replace('module', 'css_model')].size() == v.size()}
        update_key = {k for k in state_dict_2.keys() if k in model_dict.keys()}
        not_update_key = {k for k in model_dict.keys() if k not in state_dict_2.keys()}
        print("not_update_key: ", not_update_key)
        print("update_key: ", update_key)
        # pdb.set_trace()
        # print(state_dict_2.keys())
        # for k in state_dict_2.keys():
            # print(k)
            # pdb.set_trace()
        model_dict.update(state_dict_2)
        model.load_state_dict(model_dict)
        # model_dict['FC.2.weight'] - checkpoint['model']['FC.2.weight']
        # pdb.set_trace()
        # model.load_state_dict(checkpoint['model'])
    # pdb.set_trace()
    if optimizer is not None and 'join_train' in filename:
        print('load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])

def load_checkpoint_join_training_init_opti(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    # pdb.set_trace()
    if model is not None:
        model_dict = model.state_dict()
        # pdb.set_trace()
        state_dict_2 = {k.replace('module', 'css_model'):v for k,v in checkpoint['model'].items() if k.replace('module', 'css_model') in model_dict and model_dict[k].size() == v.size()}
        update_key = {k for k in state_dict_2.keys() if k in model_dict.keys()}
        print(update_key)
        # pdb.set_trace()
        # print(state_dict_2.keys())
        # for k in state_dict_2.keys():
            # print(k)
            # pdb.set_trace()
        model_dict.update(state_dict_2)
        model.load_state_dict(model_dict)
        # model_dict['FC.2.weight'] - checkpoint['model']['FC.2.weight']
        # pdb.set_trace()
        # model.load_state_dict(checkpoint['model'])
    # pdb.set_trace()
    if optimizer is not None:
        print('load optimizer')
        optimizer.load_state_dict(checkpoint['optimizer'])


def get_logger(filename):
    # Logging configuration: set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s', datefmt='%m-%d %H:%M')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("{}.log".format(filename)) 
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stderr logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger

def average_states(
    states_list: List[Dict[str, torch.Tensor]]
) -> List[Dict[str, torch.Tensor]]:
    qty = len(states_list)
    avg_state = states_list[0]
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key]

    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state

def parse_epochs(string: str) -> List[int]:
    # (a-b],(c-d]
    print(str)
    parts = string.split(',')
    res = []
    for p in parts:
        if '-' in p:
            interval = p.split('-')
            res.extend(range(int(interval[0])+1, int(interval[1])+1))
        else:
            res.append(p)
    return res

def average_checkpoints( 
    model: Module,
    models_path: str,
    epochs: str
) -> Module:
    epochs = parse_epochs(epochs)
    print(f"average model from {epochs}")
    states_dict_list = []
    for e in epochs:
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(
            models_path + f"{e}", map_location='cpu')
        copy_model.load_state_dict(checkpoint['model'])
        states_dict_list.append(copy_model.state_dict())
    avg_state_dict = average_states(states_dict_list)
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    return avg_model


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)