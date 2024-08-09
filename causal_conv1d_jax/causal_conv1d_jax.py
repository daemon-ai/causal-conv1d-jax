# -*- coding: utf-8 -*-

__all__ = ["causal_conv1d"]

from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from jax import core, dtypes, lax
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

# Import and register the CUDA extension
try:
    import causal_conv1d_jax_cuda
except ImportError:
    causal_conv1d_jax_cuda = None
else:
    for _name, _value in causal_conv1d_jax_cuda.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

#==============================================================================
def causal_conv1d_fwd(x_, weight_, bias_, activation, channel_last):
    #print("Fwd call")
    args = jnp.array([activation, channel_last], dtype=jnp.bool)
    output, = _causal_conv1d_fwd_prim.bind(x_, weight_, bias_, args)
    return output, (x_, weight_, bias_, args)

def causal_conv1d_bwd(res, grad):
    #print("Grad(out)", grad)
    #print("Bwd call", bias)
    x, weight, bias, args = res
    #args = jnp.array([activation, channel_last], dtype=jnp.bool)
    grad_input, grad_weight, grad_bias = _causal_conv1d_bwd_prim.bind(
        grad, x, weight, bias, args
    )
    return grad_input, grad_weight, grad_bias, None, None

@partial(jax.custom_vjp)
def causal_conv1d(x, weight, bias, activation=False, channel_last=False):
    output, _ = causal_conv1d_fwd(x, weight, bias, activation, channel_last)
    return output

# === Differentiation rule ===
causal_conv1d.defvjp(causal_conv1d_fwd, causal_conv1d_bwd)

# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _causal_conv1d_fwd_abstract(x, weight, bias, args):
    shape = x.shape
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    w_dtype = dtypes.canonicalize_dtype(weight.dtype)
    assert dtypes.canonicalize_dtype(weight.dtype) == dtypes.canonicalize_dtype(bias.dtype)

    assert weight.shape[-2] == shape[-2] # TODO: num kernels == 1, and dims reducd
    assert bias.shape[-1] == weight.shape[-2]
    out_shape = shape
    #print("Abs fwd", out_shape)

    return (ShapedArray(out_shape, x_dtype),)

def _causal_conv1d_bwd_abstract(grad, x, weight, bias, args):
    shape = x.shape
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    w_dtype = dtypes.canonicalize_dtype(weight.dtype)
    assert dtypes.canonicalize_dtype(weight.dtype) == dtypes.canonicalize_dtype(bias.dtype)

    assert weight.shape[-2] == shape[-2] # TODO: num kernels == 1, and dims reducd
    assert bias.shape[-1] == weight.shape[-2]
    out_shape = shape
    #print("Abs bwd", out_shape)
    # TODO: float_type = dtypes.canonicalize_dtype(jnp.float32)
    return (ShapedArray(x.shape, x_dtype), ShapedArray(weight.shape, w_dtype), ShapedArray(bias.shape, w_dtype),)

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

# Helper function
def get_op_precisions(x_np_dtype, w_np_dtype):
    prec_name = ""
    if x_np_dtype == np.float32:
        prec_name = "_f32"
    elif x_np_dtype == np.float16:
        prec_name = "_f16"
    elif x_np_dtype == jnp.bfloat16:
        prec_name = "_bf16"
    else:
        raise NotImplementedError(f"Unsupported dtype {x_np_dtype}")

    if w_np_dtype == np.float32:
        prec_name += "_f32"
    elif w_np_dtype == np.float16:
        prec_name += "_f16"
    elif w_np_dtype == jnp.bfloat16:
        prec_name += "_bf16"
    else:
        raise NotImplementedError(f"Unsupported dtype {w_np_dtype}")

    return prec_name

# Lowering:  C++ and/or CUDA interfaces to the JAX XLA backend
def _causal_conv1d_fwd_lowering(ctx, x, weight, bias, args, *, platform="cpu"):
    # Checking that input types and shape agree
    #assert x.type == weight.type # TODO: type check?

    # Extract the numpy type of the inputs
    x_aval, w_aval, _, _ = ctx.avals_in
    x_np_dtype = np.dtype(x_aval.dtype)
    w_np_dtype = np.dtype(w_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    x_dtype = mlir.ir.RankedTensorType(x.type)
    x_dims = x_dtype.shape
    w_dims = mlir.ir.RankedTensorType(weight.type).shape
    b_dims = mlir.ir.RankedTensorType(bias.type).shape
    args_dims = mlir.ir.RankedTensorType(args.type).shape
    out_shape = x_dims
    #print("OUT_SHAPE", out_shape)

    assert(mlir.ir.RankedTensorType(weight.type).element_type == mlir.ir.RankedTensorType(bias.type).element_type)

    # We dispatch a different call depending on the dtype
    op_name = platform + "_causal_conv1d_fwd" + get_op_precisions(x_np_dtype, w_np_dtype)

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        raise NotImplementedError(f"No CPU implementation!")
    elif platform == "gpu":
        if causal_conv1d_jax_cuda is None:
            raise ValueError(
                "The 'causal_conv1d' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        batch_size, dim, seqlen, width = x_dims[0], x_dims[1], x_dims[2], w_dims[-1]
        x_bs, x_c, x_l = np.prod(x_dims[1:]), np.prod(x_dims[2:]), 1
        w_c, w_width = np.prod(w_dims[1:]), 1
        out_bs, out_c, out_l = np.prod(out_shape[1:]), np.prod(out_shape[2:]), 1
        #print(x_bs, x_c, x_l, w_c, w_width, out_bs, out_c, out_l)
        opaque = causal_conv1d_jax_cuda.build_causal_conv1d_descriptor(
            batch_size, dim, seqlen, width,
            x_bs, x_c, x_l,
            w_c, w_width,
            out_bs, out_c, out_l)

        return custom_call(
            op_name,
            # Output types
            result_types=[mlir.ir.RankedTensorType.get(out_shape, x_dtype.element_type),],
            result_layouts=default_layouts(out_shape),
            # The inputs:
            operands=[x, weight, bias, args],
            operand_layouts=default_layouts(x_dims, w_dims, b_dims, args_dims),
            backend_config=opaque
        ).results

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )

def _causal_conv1d_bwd_lowering(ctx, grad_output, x, weight, bias, args, *, platform="cpu"):
    # Checking that input types and shape agree
    #assert x.type == weight.type # TODO: type check?

    grad_aval, x_aval, w_aval, _, _ = ctx.avals_in
    x_np_dtype = np.dtype(x_aval.dtype)
    w_np_dtype = np.dtype(w_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    x_dtype = mlir.ir.RankedTensorType(x.type)
    x_shape = x_dtype.shape
    dtype = mlir.ir.RankedTensorType(x.type)
    x_dims = dtype.shape
    w_dims = mlir.ir.RankedTensorType(weight.type).shape
    b_dims = mlir.ir.RankedTensorType(bias.type).shape
    grad_dims = mlir.ir.RankedTensorType(grad_output.type).shape
    args_dims = mlir.ir.RankedTensorType(args.type).shape
    out_shape = x_dims
    #print("OUT_SHAPE", out_shape)

    # We dispatch a different call depending on the dtype
    op_name = platform + "_causal_conv1d_bwd" + get_op_precisions(x_np_dtype, w_np_dtype)

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        raise NotImplementedError(f"No CPU implemetnation!")
    elif platform == "gpu":
        if causal_conv1d_jax_cuda is None:
            raise ValueError(
                "The 'causal_conv1d' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        batch_size, dim, seqlen, width = x_dims[0], x_dims[1], x_dims[2], w_dims[-1]
        x_bs, x_c, x_l = np.prod(x_dims[1:]), np.prod(x_dims[2:]), 1
        w_c, w_width = np.prod(w_dims[1:]), 1
        out_bs, out_c, out_l = np.prod(out_shape[1:]), np.prod(out_shape[2:]), 1
        #print(x_bs, x_c, x_l, w_c, w_width, out_bs, out_c, out_l)
        # TODO: custom descriptor
        opaque = causal_conv1d_jax_cuda.build_causal_conv1d_descriptor(
            batch_size, dim, seqlen, width,
            x_bs, x_c, x_l,
            w_c, w_width,
            out_bs, out_c, out_l)

        return custom_call(
            op_name,
            # Output types
            result_types=[
                mlir.ir.RankedTensorType.get(x_dims, x.type.element_type),
                mlir.ir.RankedTensorType.get(w_dims, weight.type.element_type),
                mlir.ir.RankedTensorType.get(b_dims, bias.type.element_type),
            ],
            result_layouts=default_layouts(x_dims, w_dims, b_dims),
            # The inputs:
            operands=[grad_output, x, weight, bias, args],
            operand_layouts=default_layouts(grad_dims, x_dims, w_dims, b_dims, args_dims),
            # GPU specific additional data
            backend_config=opaque
        ).results
    else:
        raise ValueError(
            "Unsupported platform; this must be either 'cpu' or 'gpu'"
        )

# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_causal_conv1d_fwd_prim = core.Primitive("causal_conv1d_fwd")
_causal_conv1d_fwd_prim.multiple_results = True
_causal_conv1d_fwd_prim.def_impl(partial(xla.apply_primitive, _causal_conv1d_fwd_prim))
_causal_conv1d_fwd_prim.def_abstract_eval(_causal_conv1d_fwd_abstract)

_causal_conv1d_bwd_prim = core.Primitive("causal_conv1d_bwd")
_causal_conv1d_bwd_prim.multiple_results = True
_causal_conv1d_bwd_prim.def_impl(partial(xla.apply_primitive, _causal_conv1d_bwd_prim))
_causal_conv1d_bwd_prim.def_abstract_eval(_causal_conv1d_bwd_abstract)


# Connect the XLA translation rules for JIT compilation
for platform in ["gpu"]:
    mlir.register_lowering(
        _causal_conv1d_fwd_prim,
        partial(_causal_conv1d_fwd_lowering, platform=platform),
        platform=platform)

    mlir.register_lowering(
        _causal_conv1d_bwd_prim,
        partial(_causal_conv1d_bwd_lowering, platform=platform),
        platform=platform)