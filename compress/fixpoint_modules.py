
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cupy

from .fixpoint import *
from . import flag
    
        
class FixpointCompressor:
    def __init__(
        self, bits=4,
        scale_method='max', scale_dims=(0,1), 
        *args, **kargs,
    ):
        '''
        bits in {2, 4, 8}
        scale_method in {'max', 'l2'}
        '''
        self.bits = bits
        self.scale_method = scale_method
        self.scale_dims = scale_dims
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=None):
        scale_shape = [micro_batch_size, seq_length, embedding_dim]
        for i in self.scale_dims:
            scale_shape[i] = 1
        
        # Communication Buffers
        self.buffers = [
            (
                torch.zeros((micro_batch_size, seq_length, embedding_dim * self.bits // 8), 
                            requires_grad=False, device=device, dtype=torch.uint8),
                torch.zeros(scale_shape, 
                            requires_grad=False, device=device, dtype=torch.float16),
            ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # Communication Buffers during Warmup (w/o compression)
        self.warmup_buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
    def compress(self, x):
        if self.bits == 8:
            x, scale = compress_8bit(x, scale_method=self.scale_method, scale_dims=self.scale_dims)
            return x, scale
        elif self.bits == 4:
            x, scale = compress_4bit(x, scale_method=self.scale_method, scale_dims=self.scale_dims)
            return x, scale
        elif self.bits == 2:
            x, scale = compress_2bit(x, scale_method=self.scale_method, scale_dims=self.scale_dims)
            return x, scale
        
        raise Exception(f'no solution to bits {self.bits}')
        
    def decompress(self, x):
        if self.bits == 8:
            ret = decompress_8bit(*x)
            return ret
        elif self.bits == 4:
            ret = decompress_4bit(*x)
            return ret
        elif self.bits == 2:
            ret = decompress_2bit(*x)
            return ret
        
        raise Exception(f'no solution to bits {self.bits}')
        
        
    def compress_send(self, x, i_micro_batch, comm, dst, stream):
        if not flag.FLAG_DISABLE_COMPRESSION:
            with stream:
                _data = self.compress(x)
            for _x in _data:
                comm.send(_x, dst=dst, stream=stream)
        else:
            comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        if not flag.FLAG_DISABLE_COMPRESSION:
            recv_buffer = self.buffers[i_micro_batch]
            for _recv_buffer in recv_buffer:
                comm.recv(_recv_buffer, src=src, stream=stream)
            with stream:
                x = self.decompress(recv_buffer)
            return x
        else:
            recv_buffer = self.warmup_buffers[i_micro_batch]
            comm.recv(recv_buffer, src=src, stream=stream)
            return recv_buffer
    
    
    
class FixpointFlexibleCompressor(FixpointCompressor):
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=None):
        self.activ_shape = (micro_batch_size, seq_length, embedding_dim)
        scale_shape = [micro_batch_size, seq_length, embedding_dim]
        for i in self.scale_dims:
            scale_shape[i] = 1
        self.scale_shape = scale_shape
        
        # Communication Buffers
        _x = torch.randn(self.activ_shape).to(device)
        _a, _s = compress_flexible_nbits(_x, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
        self.buffers = [
            (
                torch.zeros(_a.shape, requires_grad=False, device=device, dtype=torch.uint8),
                torch.zeros(_s.shape, requires_grad=False, device=device, dtype=torch.float16),
            ) for _ in range(batch_size//micro_batch_size)
        ]
        
        # Communication Buffers during Warmup (w/o compression)
        self.warmup_buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
    def compress(self, x):
        return compress_flexible_nbits(x, self.bits, scale_method=self.scale_method, scale_dims=self.scale_dims)
        
    def decompress(self, x):
        return decompress_flexible_nbits(*x, self.bits, self.activ_shape)
        

            