import torch
from .grad_scalar import *



def _zero_grad_group(group, set_to_none):
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(this, that):
    for this_, that_ in zip(this, that):
        that_.copy_(this_)


class Fp16Optimizer:
    def __init__(self, optimizer, grad_scaler):
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler

        if self.grad_scaler:
            self.found_inf = torch.cuda.FloatTensor([0.0])

        self._dummy_overflow_buf = torch.cuda.IntTensor([0])

        # Note that the model should first be cast to fp16 before passing to the optimizer.
        self.float16_groups = []
        self.fp32_from_float16_groups = []

        # For all the groups in the original optimizer:
        for param_group in self.optimizer.param_groups:
            float16_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    # float16 params:
                    assert param.type() == 'torch.cuda.HalfTensor'
                    float16_params_this_group.append(param)
                    # Create a copy
                    optimizer_param = param.detach().clone().float()
                    # Replace the optimizer params with the new fp32 copy.
                    param_group['params'][i] = optimizer_param
                    fp32_from_float16_params_this_group.append(optimizer_param)
                    # Reset existing state dict key to the new optimizer param.
                    if param in self.optimizer.state:
                        self.optimizer.state[optimizer_param] = self.optimizer.state.pop(param)

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)

        # Leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def zero_grad(self, set_to_none=True):
        for group in self.float16_groups:
            _zero_grad_group(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            _zero_grad_group(group, set_to_none)

    def get_loss_scale(self):
        return self.grad_scaler.scale

    def _copy_model_grads_to_optimizer_grads(self):
        # This only needs to be done for the float16 group.
        for model_group, optimizer_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, optimizer_param in zip(model_group, optimizer_group):
                if model_param.grad is not None:
                    optimizer_param.grad = model_param.grad.float()
                # Safe to deallocate model's grad/optimizer_grad after copying.
                # (If using contiguous buffers, optimizer_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None

    def _unscale_optimizer_grads_and_check_for_nan(self):
        optimizer_grads = []
        # fp32 params fromm float16 ones.
        for optimizer_group in self.fp32_from_float16_groups:
            for optimizer_param in optimizer_group:
                if optimizer_param.grad is not None:
                    optimizer_grads.append(optimizer_param.grad.data)
        # Reset found inf.
        self.found_inf.fill_(0.0)
        # Unscale and set found inf/nan
        torch._amp_foreach_non_finite_check_and_unscale_(optimizer_grads, self.found_inf, self.grad_scaler.inv_scale)
        # Check for nan.
        found_inf_flag = (self.found_inf.item() > 0)
        return found_inf_flag

    def _get_model_and_optimizer_params_data_float16(self):
        model_data = []
        optimizer_data = []
        for model_group, optimizer_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, optimizer_param in zip(model_group, optimizer_group):
                model_data.append(model_param.data)
                optimizer_data.append(optimizer_param.data)
        return model_data, optimizer_data

    def _copy_optimizer_params_to_model_params(self):
        # Only needed for the float16 params.
        model_data, optimizer_data = self._get_model_and_optimizer_params_data_float16()
        _multi_tensor_copy_this_to_that(this=optimizer_data, that=model_data)

    def _copy_model_params_to_optimizer_params(self):
        # Only needed for the float16 params.
        model_data, optimizer_data = self._get_model_and_optimizer_params_data_float16()
        _multi_tensor_copy_this_to_that(this=model_data, that=optimizer_data)

    def reload_model_params(self):
        self._copy_model_params_to_optimizer_params()

    @torch.no_grad()
    def step(self):
        self._copy_model_grads_to_optimizer_grads()

        found_inf_flag = self._unscale_optimizer_grads_and_check_for_nan()
        self.grad_scaler.update(found_inf_flag)

        # If we found inf/nan, skip the update.
        if found_inf_flag:
            print("!!! Warning: find inf in fp16 optimizer-step() !!!")
            # return False

        # Step the optimizer.
        self.optimizer.step()

        self._copy_optimizer_params_to_model_params()
        # Successful update.
        return True


def get_fp16_optimizer(args, optimizer):
    assert args.fp16 is not None
    if args.loss_scale:
        grad_scaler = ConstantGradScaler(args.loss_scale)
    else:
        grad_scaler = DynamicGradScaler(
            initial_scale=args.initial_loss_scale,
            min_scale=args.min_loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=args.loss_scale_window,
            hysteresis=args.hysteresis)
    return Fp16Optimizer(optimizer, grad_scaler)


