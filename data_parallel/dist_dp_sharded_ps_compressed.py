import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import math
from comm.comm_utils import *
from .flatten_utils import flatten_params, flatten_tensors
from compress.fixpoint import *
from compress import flag


@torch.no_grad()
def get_exp_avgs(self):
    assert isinstance(self, torch.optim.AdamW)
    
    exp_avgs = []

    for group in self.param_groups:
        beta1, beta2 = group['betas']

        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError('AdamW does not support sparse gradients')
                
            grad = p.grad
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg = state['exp_avg']
            exp_avgs.append(exp_avg)
    
    return exp_avgs


@torch.no_grad()
def step_update_exp_avg(self):
    assert isinstance(self, torch.optim.AdamW)

    for group in self.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

            state = self.state[p]

            assert len(state) != 0

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            

@torch.no_grad()
def step_update(self, freeze=False):
    assert isinstance(self, torch.optim.AdamW)

    for group in self.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

            state = self.state[p]

            assert len(state) != 0

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = group["betas"]

            state["step"] += 1

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
#             exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            if not freeze:
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

            step_size = group["lr"]
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(exp_avg, denom, value=-step_size)

            if group["weight_decay"] > 0.0:
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
            

def step_adamw(self, closure=None, freeze=False):
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
            if not freeze:
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            denom = exp_avg_sq.sqrt().add_(group["eps"])

            step_size = group["lr"]
            bias_correction1 = 1.0 - beta1 ** state["step"]
            bias_correction2 = 1.0 - beta2 ** state["step"]
            step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(exp_avg, denom, value=-step_size)

            if group["weight_decay"] > 0.0:
                p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

    return loss
            


class ShardedPSDPCompressed:
    def __init__(self, args, device, module: torch.nn.Module, optimizer: torch.optim.Optimizer = None, flatten=True):
        self.dp_bits = args.dp_bits
        self.flatten = flatten
        self.global_rank = args.rank
        self.dp_group_size = args.data_group_size
        self.enable_tidy_profiling = (args.profiling == 'tidy_profiling')
        self.dp_comm = get_data_parallel_comm()
        self.dp_rank = get_data_parallel_rank()
        self.dp_comm_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_optim_comp_stream = torch.cuda.default_stream(device=device)
        self.backward_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.sync_gradients_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)
        self.optimizer_step_ready_event = torch.cuda.Event(enable_timing=self.enable_tidy_profiling, blocking=False)

        self.module = module
        assert optimizer is not None
        self.optimizer = optimizer
        num_paras, element_size = self._compute_total_para_num()
        print("Total number of parameters: {}, element size: {}, total size {} MB."
              .format(num_paras, element_size, num_paras * element_size // 1024 // 1024))

        assert self.flatten
#         self.para = list(self.module.parameters())
        self.flatten_para = flatten_params(self.module.parameters(), self.dp_group_size)
        print("Flattened parameter number: {}, element size: {}."
              .format(self.flatten_para.data.numel(), self.flatten_para.data.element_size()))
        print("Flattened parameter grad number: {}, element size: {}."
              .format(self.flatten_para.grad.numel(), self.flatten_para.grad.element_size()))
        
        exp_avgs = get_exp_avgs(self.optimizer)
        self.flatten_exp_avgs = flatten_tensors(exp_avgs, self.dp_group_size*512*4)

        self.grad_buffer_non_compressed = self._declare_grad_buffer()
        
        self.exp_avg_buffer, self.exp_avg_worker_errors, self.exp_avg_server_error \
            = self._declare_exp_avgs_buffer()
        
        
        if self.enable_tidy_profiling:
            self.global_rank = args.rank
            self.init_event = None
            self.init_time_stamp = None

            assert self.flatten
            self.sync_gradients_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.optimizer_step_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            
            self.gather_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.sync_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.gather_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.sync_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            
            self.worker_compress_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.server_compress_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.worker_compress_end_event = torch.cuda.Event(enable_timing=True, blocking=False)
            self.server_compress_end_event = torch.cuda.Event(enable_timing=True, blocking=False)

    def _compute_total_para_num(self):
        total_count = 0
        element_size = 0
        for para in self.module.parameters():
            # print("Parameter: ", para.data.shape)
            total_count += torch.numel(para.data)
            element_size = para.element_size()
        return total_count, element_size

    def _declare_grad_buffer(self):
        assert self.flatten_para.data.numel() % self.dp_group_size == 0
        chunk_size = self.flatten_para.data.numel() // self.dp_group_size
        bits = self.dp_bits
        _data = torch.zeros(chunk_size, device=self.flatten_para.device, dtype=self.flatten_para.dtype)

        grad_buffer_non_compressed = [
            torch.zeros_like(_data) for i in range(self.dp_group_size)
        ]

        return grad_buffer_non_compressed #, grad_buffer, worker_errors, server_error

    def _declare_exp_avgs_buffer(self):
        assert self.flatten_exp_avgs.data.numel() % self.dp_group_size == 0
        chunk_size = self.flatten_exp_avgs.data.numel() // self.dp_group_size
        bits = self.dp_bits
        _data = torch.zeros(chunk_size, device=self.flatten_exp_avgs.device, dtype=self.flatten_exp_avgs.dtype)
#         _data_compressed = compress_flexible_nbits(_data, bits=bits, scale_dims=tuple())
        _data_compressed = compress_flexible_nbits_by_bucket(
            _data, bits=bits, bucket_size=512)
        grad_buffer = [
            (torch.zeros_like(_data_compressed[0]),
             torch.zeros_like(_data_compressed[1])) for i in range(self.dp_group_size)
        ]
        worker_errors = [
            torch.zeros_like(_data).half() for i in range(self.dp_group_size)
        ]
        server_error = torch.zeros_like(_data).half()
        return grad_buffer, worker_errors, server_error

    def profile_mark_sync_grad_start(self):
        if self.enable_tidy_profiling:
            self.dp_comm_stream.record_event(self.sync_gradients_start_event)

    def profile_mark_allreduce_end(self):
        pass

    def profile_mark_optimizer_step_start(self):
        if self.enable_tidy_profiling:
            self.torch_optim_comp_stream.record_event(self.optimizer_step_start_event)

    def _sync_gradients(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            assert self.flatten
            self.profile_mark_sync_grad_start()
            if flag.FLAG_DISABLE_COMPRESSION:
                self.dp_comm.all_reduce_opt(
                    self.flatten_para.grad, self.grad_buffer_non_compressed, 
                    stream=cupy_dp_stream)
            else:
                print('Compression is enabled now!')
                self.dp_comm.all_reduce_opt_compressed(
                    self.flatten_para.grad, self.grad_buffer, 
                    self.worker_errors, self.server_error, 
                    stream=cupy_dp_stream, bits=self.dp_bits)
            self.profile_mark_allreduce_end()
            self.dp_comm_stream.record_event(self.sync_gradients_ready_event)
            
    def _sync_exp_avgs(self):
        with torch.cuda.stream(self.dp_comm_stream):
            cupy_dp_stream = cupy.cuda.ExternalStream(self.dp_comm_stream.cuda_stream)
            self.dp_comm_stream.wait_event(self.backward_ready_event)
            assert self.flatten
            self.profile_mark_sync_grad_start()
            if flag.FLAG_DISABLE_COMPRESSION:
                # send grad
                self.dp_comm.all_reduce_opt(
                    self.flatten_para.grad, self.grad_buffer_non_compressed, 
                    stream=cupy_dp_stream, caller=self)
                step_update_exp_avg(self.optimizer)
            else:
                # send compressed exp avg
                print('Compression is enabled now!')
                step_update_exp_avg(self.optimizer)
                self.dp_comm.all_reduce_opt_compressed(
                    self.flatten_exp_avgs.data, self.exp_avg_buffer, 
                    self.exp_avg_worker_errors, self.exp_avg_server_error, 
                    stream=cupy_dp_stream, bits=self.dp_bits, caller=self)
            self.profile_mark_allreduce_end()
            self.dp_comm_stream.record_event(self.sync_gradients_ready_event)

    def optimizer_step(self):
#         self._sync_gradients()
        self._sync_exp_avgs()
        with torch.cuda.stream(self.torch_optim_comp_stream):
            self.torch_optim_comp_stream.wait_event(self.sync_gradients_ready_event)
            self.profile_mark_optimizer_step_start()
#             self.optimizer.step()
#             step_adamw(self.optimizer, freeze=(not flag.FLAG_DISABLE_COMPRESSION))
            step_update(self.optimizer, freeze=(not flag.FLAG_DISABLE_COMPRESSION))
            self.torch_optim_comp_stream.record_event(self.optimizer_step_ready_event)

    def set_time_stamp(self, init_time_stamp, init_event):
        self.init_event = init_event
        self.init_time_stamp = init_time_stamp

    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3

    def profiling_data_parallel(self, init_time_stamp, init_event):
        self.set_time_stamp(init_time_stamp, init_event)
        profiling_log = []

        assert self.flatten
        allreduce_slot = self.sync_gradients_start_event.elapsed_time(self.sync_gradients_ready_event)*1e+3
        allreduce_log = {"name": "opt_shardedPS_sync", "ph": "X", "pid": self.global_rank, "tid": "7. optimizer-comm",
                         "ts": self.get_ts(self.sync_gradients_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)

        optimizer_slot = self.optimizer_step_start_event.elapsed_time(self.optimizer_step_ready_event) * 1e+3
        optimizer_log = {"name": "opt_comp", "ph": "X", "pid": self.global_rank, "tid": "8. optimizer-comp",
                         "ts": self.get_ts(self.optimizer_step_start_event), "dur": optimizer_slot, "cname": "bad"}
        # print(optimizer_log)
        profiling_log.append(optimizer_log)
        
        
        allreduce_slot = self.gather_start_event.elapsed_time(self.gather_end_event)*1e+3
        allreduce_log = {"name": "gather grads", "ph": "X", "pid": self.global_rank, "tid": "9. optimizer-comm",
                         "ts": self.get_ts(self.gather_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        allreduce_slot = self.sync_start_event.elapsed_time(self.sync_end_event)*1e+3
        allreduce_log = {"name": "distribute grads", "ph": "X", "pid": self.global_rank, "tid": "10. optimizer-comm",
                         "ts": self.get_ts(self.sync_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        allreduce_slot = self.worker_compress_start_event.elapsed_time(self.worker_compress_end_event)*1e+3
        allreduce_log = {"name": "worker compress", "ph": "X", "pid": self.global_rank, "tid": "11. optimizer-comm",
                         "ts": self.get_ts(self.worker_compress_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        allreduce_slot = self.server_compress_start_event.elapsed_time(self.server_compress_end_event)*1e+3
        allreduce_log = {"name": "server compress", "ph": "X", "pid": self.global_rank, "tid": "12. optimizer-comm",
                         "ts": self.get_ts(self.server_compress_start_event),
                         "dur": allreduce_slot, "cname": "cq_build_passed",
                         "args": {'para': 'flattened_grad', 'size': self.flatten_para.grad.numel()}}
        # print(allreduce_log)
        profiling_log.append(allreduce_log)
        
        return profiling_log
