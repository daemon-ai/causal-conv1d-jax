import torch
import torch.nn.functional as F

# https://github.com/Dao-AILab/causal-conv1d/blob/a40ff7436f78a0a20da593e87dfd3e3673c0819e/causal_conv1d/causal_conv1d_interface.py#L49
def causal_conv1d_ref(x, weight, bias=None, activation=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    out = out[..., :seqlen]
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)