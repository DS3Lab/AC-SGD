import torch
import cupy
import numpy as np

from .utils import *


def _rounding(x, stochastic=False, minimum_stochastic_distance=0.2):
    if stochastic:
        x_floor = x.floor()
        th = x - x_floor
        if minimum_stochastic_distance > 0:
            th[th<minimum_stochastic_distance] = 0.
            th[th>1-minimum_stochastic_distance] = 1.
        pr = torch.rand_like(x)
        x_floor += (pr < th)
        return x_floor
    else:
        return x.round()


def _compress_nbits(x, bits, scale_method='max', scale_dims=(0,1)):
    
    fbits = bits - 1
    
    if scale_method == 'max':
        # issue: sensitive to outlier points
        scale = x.abs().amax(scale_dims, keepdims=True)
    elif scale_method == 'l2':
        # ~95% confidence interval for normal distribution
        scale = x.pow(2).mean(scale_dims, keepdims=True).sqrt() * 2 
    else:
        raise Exception('unkonwn scale method.')
    # fp16 should be enough
    scale = scale.half()
    x = x / (scale + 1e-6)
    
    x = x.ldexp(torch.tensor(fbits))
    clip_min = -(1<<fbits)
    clip_max = (1<<fbits)-1

    x = _rounding(x)
    x = x.clip(clip_min, clip_max)
    
    x = x - clip_min
    x = x.type(torch.uint8)
    
    return x, scale


def _decompress_nbits(x, scale, bits):
    
    fbits = bits - 1
    
    clip_min = -(1<<fbits)
    clip_max = (1<<fbits)-1
    
    x = x.float() + clip_min
    
    x = x / (clip_max+1) * scale
    
    return x


def compress_8bit(x, scale_method='max', scale_dims=(0,1)):

    x, scale = _compress_nbits(x, bits=8, scale_method=scale_method, scale_dims=scale_dims)
    
    return x, scale


def decompress_8bit(x, scale):
    
    x = _decompress_nbits(x, scale, bits=8)
    
    return x

def compress_4bit(x, scale_method='max', scale_dims=(0,1)):

    x, scale = _compress_nbits(x, bits=4, scale_method=scale_method, scale_dims=scale_dims)
    
    x0, x1 = x.chunk(2, -1)
    x = (x0 << 4) + x1
    
    return x, scale


def decompress_4bit(x, scale):
    
    bitmask = 15
    
    x0 = (x >> 4)
    x1 = (x & bitmask)
    
    x = torch.cat([x0, x1], -1)
    
    x = _decompress_nbits(x, scale, bits=4)
    
    return x


def compress_2bit(x, scale_method='max', scale_dims=(0,1)):

    x, scale = _compress_nbits(x, bits=2, scale_method=scale_method, scale_dims=scale_dims)
    
    x0, x1, x2, x3 = x.chunk(4, -1)
    x = (x0 << 6) + (x1 << 4) + (x2 << 2) + x3
    
    return x, scale


def decompress_2bit(x, scale):
    
    bitmask = 3
    
    x0 = (x >> 6)
    x1 = (x >> 4) & bitmask
    x2 = (x >> 2) & bitmask
    x3 = x & bitmask
    x = torch.cat([x0, x1, x2, x3], -1)
    
    x = _decompress_nbits(x, scale, bits=2)
    
    return x



def compress_flexible_nbits(x, bits, scale_method='max', scale_dims=(0,1)):
    # support any bits
    # CUDA only
    
    x, scale = _compress_nbits(x, bits=bits, scale_method=scale_method, scale_dims=scale_dims)
    
    x = pack_low_bit_tensor(x, bits)
    
    return x, scale


def decompress_flexible_nbits(x, scale, bits, original_shape):
    # support any bits, but need to know original_shape
    # CUDA only
    
    x = unpack_low_bit_tensor(x, bits, original_shape)
    
    x = _decompress_nbits(x, scale, bits=bits)
    
    return x



def compress_nbits(x, bits, scale_method='max', scale_dims=(0,1)):
    if bits == 8:
        x, scale = compress_8bit(x, scale_method=scale_method, scale_dims=scale_dims)
    elif bits == 4:
        x, scale = compress_4bit(x, scale_method=scale_method, scale_dims=scale_dims)
    elif bits == 2:
        x, scale = compress_2bit(x, scale_method=scale_method, scale_dims=scale_dims)
    
    return x, scale


def decompress_nbits(x, scale, bits):
    if bits == 8:
        y = decompress_8bit(x, scale)
    elif bits == 4:
        y = decompress_4bit(x, scale)
    elif bits == 2:
        y = decompress_2bit(x, scale)
    
    return y




def _compress_nbits_by_bucket(x, bits, scale_method='max', bucket_size=512):
    
    fbits = bits - 1
    
    x = x.view(bucket_size, -1)
    
    if scale_method == 'max':
        # issue: sensitive to outlier points
        scale = x.abs().amax([0], keepdims=True)
    elif scale_method == 'l2':
        # ~95% confidence interval for normal distribution
        scale = x.pow(2).mean([0], keepdims=True).sqrt() * 2 
    else:
        raise Exception('unkonwn scale method.')
    # fp16 should be enough
    scale = scale.half()
    x = x / (scale + 1e-6)
    
    x = x.ldexp(torch.tensor(fbits))
    clip_min = -(1<<fbits)
    clip_max = (1<<fbits)-1

    x = _rounding(x)
    x = x.clip(clip_min, clip_max)
    
    x = x - clip_min
    x = x.type(torch.uint8)
    
    return x, scale


def compress_flexible_nbits_by_bucket(x, bits, scale_method='max', bucket_size=512):
    # support any bits
    # CUDA only
    
    x, scale = _compress_nbits_by_bucket(x, bits=bits, scale_method=scale_method, bucket_size=bucket_size)
    
    x = pack_low_bit_tensor(x, bits)
    
    return x, scale


def decompress_flexible_nbits_by_bucket(x, scale, bits, original_shape, bucket_size=512):
    # support any bits, but need to know original_shape
    # CUDA only
    
    x = unpack_low_bit_tensor(x, bits, original_shape)

    x = x.view(bucket_size, -1)
    x = _decompress_nbits(x, scale, bits=bits)
    x = x.view(original_shape)
    
    return x