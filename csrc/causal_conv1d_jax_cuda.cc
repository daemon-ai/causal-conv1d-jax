// This file defines the Python interface to the XLA custom call implemented on the GPU.
// Like in cpu_ops.cc, we export a separate capsule for each supported dtype, but we also
// include one extra method "build_kepler_descriptor" to generate an opaque representation
// of the problem size that will be passed to the op. The actually implementation of the
// custom call can be found in kernels.cc.cu.

#include "kernels.h"
#include "pybind11_kernel_helpers.h"

using namespace causal_conv1d_jax;

namespace {
pybind11::dict Registrations() {
  pybind11::dict dict;
  // Forward
  //dict["gpu_causal_conv1d_fwd_f64"] = EncapsulateFunction(gpu_causal_conv1d_fwd_f64);
  dict["gpu_causal_conv1d_fwd_f32_f32"] = EncapsulateFunction(gpu_causal_conv1d_fwd_f32);
  dict["gpu_causal_conv1d_fwd_f16_f16"] = EncapsulateFunction(gpu_causal_conv1d_fwd_f16);
  dict["gpu_causal_conv1d_fwd_bf16_bf16"] = EncapsulateFunction(gpu_causal_conv1d_fwd_bf16);

  dict["gpu_causal_conv1d_fwd_f32_f16"] = EncapsulateFunction(gpu_causal_conv1d_fwd_f32_f16);
  dict["gpu_causal_conv1d_fwd_f32_bf16"] = EncapsulateFunction(gpu_causal_conv1d_fwd_f32_bf16);

  dict["gpu_causal_conv1d_fwd_f16_f32"] = EncapsulateFunction(gpu_causal_conv1d_fwd_f16_f32);
  dict["gpu_causal_conv1d_fwd_f16_bf16"] = EncapsulateFunction(gpu_causal_conv1d_fwd_f16_bf16);

  dict["gpu_causal_conv1d_fwd_bf16_f32"] = EncapsulateFunction(gpu_causal_conv1d_fwd_bf16_f32);
  dict["gpu_causal_conv1d_fwd_bf16_f16"] = EncapsulateFunction(gpu_causal_conv1d_fwd_bf16_f16);

  // Backward
  //dict["gpu_causal_conv1d_bwd_f64"] = EncapsulateFunction(gpu_causal_conv1d_bwd_f64);
  dict["gpu_causal_conv1d_bwd_f32_f32"] = EncapsulateFunction(gpu_causal_conv1d_bwd_f32);
  dict["gpu_causal_conv1d_bwd_f16_f16"] = EncapsulateFunction(gpu_causal_conv1d_bwd_f16);
  dict["gpu_causal_conv1d_bwd_bf16_bf16"] = EncapsulateFunction(gpu_causal_conv1d_bwd_bf16);

  dict["gpu_causal_conv1d_bwd_f32_f16"] = EncapsulateFunction(gpu_causal_conv1d_bwd_f32_f16);
  dict["gpu_causal_conv1d_bwd_f32_bf16"] = EncapsulateFunction(gpu_causal_conv1d_bwd_f32_bf16);

  dict["gpu_causal_conv1d_bwd_f16_f32"] = EncapsulateFunction(gpu_causal_conv1d_bwd_f16_f32);
  dict["gpu_causal_conv1d_bwd_f16_bf16"] = EncapsulateFunction(gpu_causal_conv1d_bwd_f16_bf16);

  dict["gpu_causal_conv1d_bwd_bf16_f32"] = EncapsulateFunction(gpu_causal_conv1d_bwd_bf16_f32);
  dict["gpu_causal_conv1d_bwd_bf16_f16"] = EncapsulateFunction(gpu_causal_conv1d_bwd_bf16_f16);
  // TODO: list others
  return dict;
}

PYBIND11_MODULE(causal_conv1d_jax_cuda, m) {
  m.def("registrations", &Registrations);
  m.def("build_causal_conv1d_descriptor",[](
    uint32_t batch_size, uint32_t dim, uint32_t seqlen, uint32_t width,
    uint32_t x_bs, uint32_t x_c, uint32_t x_l,
    uint32_t w_c, uint32_t w_width,
    uint32_t out_bs, uint32_t out_c, uint32_t out_l) {
          return PackDescriptor(CausalConv1dDescriptor{
            batch_size, dim, seqlen, width, x_bs, x_c, x_l, w_c, w_width, out_bs, out_c, out_l
            });
    });
}
}  // namespace