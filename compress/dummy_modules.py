
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cupy


class NoCompression:
    def __init__(self, *args, **kargs):
        pass
        
    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=torch.float32):
        self.buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]
        
    def compress(self, x):
        return x
        
    def decompress(self, x):
        return x
        
    def compress_send(self, x, i_micro_batch, comm, dst, stream):
        comm.send(x, dst=dst, stream=stream)
        
    def recv_decompress(self, i_micro_batch, comm, src, stream):
        recv_buffer = self.buffers[i_micro_batch]
        comm.recv(recv_buffer, src=src, stream=stream)
        return recv_buffer
    