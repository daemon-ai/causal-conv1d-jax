// This file contains the GPU implementation of our op. It's a pretty typical CUDA kernel
// and I make no promises about the quality of the code or the choices made therein, but
// it should get the point accross.

#include "kernels.h"
#include "kernel_helpers.h"
#include "cuda_src/causal_conv1d.h"

#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

#include "cuda_src/causal_conv1d_common.h"

namespace causal_conv1d_jax {

namespace {

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

#include "cuda_src/causal_conv1d_fwd.cu"

void set_params(ConvParamsBase &params, const CausalConv1dDescriptor &d){
  params.batch = d.batch;
  params.dim = d.dim;
  params.seqlen = d.seqlen;
  params.width = d.width;

  params.x_batch_stride = d.x_batch_stride;
  params.x_c_stride = d.x_c_stride;
  params.x_l_stride = d.x_l_stride;
  params.weight_c_stride = d.weight_c_stride;
  params.weight_width_stride = d.weight_width_stride;
  params.out_batch_stride = d.out_batch_stride;
  params.out_c_stride = d.out_c_stride;
  params.out_l_stride = d.out_l_stride;
}


#include "cuda_src/causal_conv1d_bwd.cu"

void set_bwd_params(ConvParamsBwd &params, const CausalConv1dDescriptor &d){
  params.batch = d.batch;
  params.dim = d.dim;
  params.seqlen = d.seqlen;
  params.width = d.width;

  params.x_batch_stride = d.x_batch_stride;
  params.x_c_stride = d.x_c_stride;
  params.x_l_stride = d.x_l_stride;
  params.weight_c_stride = d.weight_c_stride;
  params.weight_width_stride = d.weight_width_stride;
  params.out_batch_stride = d.out_batch_stride;
  params.out_c_stride = d.out_c_stride;
  params.out_l_stride = d.out_l_stride;

  // grad params
  params.dx_batch_stride = d.x_batch_stride;
  params.dx_c_stride = d.x_c_stride;
  params.dx_l_stride = d.x_l_stride;
  params.dweight_c_stride = d.weight_c_stride;
  params.dweight_width_stride = d.weight_width_stride;
  params.dout_batch_stride = d.out_batch_stride;
  params.dout_c_stride = d.out_c_stride;
  params.dout_l_stride = d.out_l_stride;
}

// === DEBUG ===
template <typename T>
__global__ void set_val(int size, T val, T* out){
  out[size] = val;
}

template <typename T>
void debug_params(ConvParamsBase &params, T* out){
  set_val<<<1,1>>>(0, (float)params.batch, out);
  set_val<<<1,1>>>(1, (float)params.dim, out);
  set_val<<<1,1>>>(2, (float)params.seqlen, out);
  set_val<<<1,1>>>(3, (float)params.width, out);

  set_val<<<1,1>>>(0, (float)params.x_batch_stride, out);
  set_val<<<1,1>>>(1, (float)params.x_c_stride, out);
  set_val<<<1,1>>>(2, (float)params.x_l_stride, out);
  set_val<<<1,1>>>(3, (float)params.weight_c_stride, out);
  set_val<<<1,1>>>(4, (float)params.weight_width_stride, out);
  set_val<<<1,1>>>(6, (float)params.out_batch_stride, out);
  set_val<<<1,1>>>(7, (float)params.out_c_stride, out);
  set_val<<<1,1>>>(8, (float)params.out_l_stride, out);
}

template <typename input_t, typename weight_t>
inline void apply_causal_conv1d_fwd(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
  // TODO: const?
  const CausalConv1dDescriptor &d = *UnpackDescriptor<CausalConv1dDescriptor>(opaque, opaque_len);

  // TODO: const for the first 3
  input_t *x = reinterpret_cast<input_t *>(buffers[0]);
  weight_t *weight = reinterpret_cast<weight_t *>(buffers[1]);
  weight_t *bias = reinterpret_cast<weight_t *>(buffers[2]);
  bool* args = reinterpret_cast<bool *>(buffers[3]);
  input_t *out = reinterpret_cast<input_t *>(buffers[4]);
  bool is_channel_last=false;

  ConvParamsBase params;
  set_params(params, d);
  cudaMemcpy(&(params.silu_activation), args, sizeof(bool), cudaMemcpyDeviceToHost);
  //cudaMemcpy(&(is_channel_last), args+1, sizeof(bool), cudaMemcpyDeviceToHost);

  params.x_ptr = x;
  params.weight_ptr = weight;
  params.bias_ptr = bias;
  params.out_ptr = out;

  //[DEBUG]
  //debug_params<T>(params, out);
  //return;

  //if(is_channel_last){
  //  params.x_l_stride = d.x_c_stride;
  //  params.x_c_stride = d.x_l_stride;
  //  causal_conv1d_channellast_fwd_cuda<input_t, weight_t>(params, stream);
  //}
  //else{
  //  causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
  //}
  causal_conv1d_fwd_cuda<input_t, weight_t>(params, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ThrowIfError(cudaGetLastError());
}

template <typename input_t, typename weight_t>
inline void apply_causal_conv1d_bwd(cudaStream_t stream, void **buffers, const char *opaque,
                         std::size_t opaque_len) {
    const CausalConv1dDescriptor &d = *UnpackDescriptor<CausalConv1dDescriptor>(opaque, opaque_len);

    input_t *grad_out = reinterpret_cast<input_t *>(buffers[0]);
    input_t *x = reinterpret_cast<input_t *>(buffers[1]);
    weight_t *weight = reinterpret_cast<weight_t *>(buffers[2]);
    weight_t *bias = reinterpret_cast<weight_t *>(buffers[3]);
    bool* args = reinterpret_cast<bool *>(buffers[4]);
    bool is_channel_last=false;

    input_t *grad_x = reinterpret_cast<input_t *>(buffers[5]);
    float *grad_weight = reinterpret_cast<float *>(buffers[6]);
    float *grad_bias = reinterpret_cast<float *>(buffers[7]);

    cudaMemset(grad_x, 0, sizeof(input_t)*d.batch*d.dim*d.seqlen);
    cudaMemset(grad_weight, 0, sizeof(float)*d.dim*d.width);
    cudaMemset(grad_bias, 0, sizeof(float)*d.dim);

    ConvParamsBwd params;
    set_bwd_params(params, d);
    cudaMemcpy(&(params.silu_activation), args, sizeof(bool), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&(is_channel_last), args+1, sizeof(bool), cudaMemcpyDeviceToHost);

    params.x_ptr = x;
    params.weight_ptr = weight;
    params.bias_ptr = bias;

    params.dout_ptr = grad_out;
    params.dx_ptr = grad_x;
    params.dweight_ptr = grad_weight;
    params.dbias_ptr = grad_bias;

    //if(is_channel_last){
    //  params.x_l_stride = d.x_c_stride;
    //  params.x_c_stride = d.x_l_stride;
    //  causal_conv1d_channellast_bwd_cuda<input_t, weight_t>(params, stream);
    //}
    //else{
    //  causal_conv1d_bwd_cuda<input_t, weight_t>(params, stream);
    //}
    causal_conv1d_bwd_cuda<input_t, weight_t>(params, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    ThrowIfError(cudaGetLastError());
}

}  // namespace

// === Forward functions ===
//void gpu_causal_conv1d_fwd_f64(cudaStream_t stream, void** buffers, const char* opaque,
//  std::size_t opaque_len){
//    apply_causal_conv1d_fwd<double, double>(stream, buffers, opaque, opaque_len);
//}
void gpu_causal_conv1d_fwd_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_fwd<float, float>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_fwd_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_fwd<__half, __half>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_fwd_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_fwd<__nv_bfloat16, __nv_bfloat16>(stream, buffers, opaque, opaque_len);
}

void gpu_causal_conv1d_fwd_f32_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_fwd<float, __half>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_fwd_f32_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_fwd<float, __nv_bfloat16>(stream, buffers, opaque, opaque_len);
}

void gpu_causal_conv1d_fwd_f16_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_fwd<__half, float>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_fwd_f16_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_fwd<__half, __nv_bfloat16>(stream, buffers, opaque, opaque_len);
}

void gpu_causal_conv1d_fwd_bf16_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_fwd<__nv_bfloat16, float>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_fwd_bf16_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_fwd<__nv_bfloat16, __half>(stream, buffers, opaque, opaque_len);
}

// === Backward function ===
//void gpu_causal_conv1d_bwd_f64(cudaStream_t stream, void** buffers, const char* opaque,
//  std::size_t opaque_len){
//    apply_causal_conv1d_bwd<double, double>(stream, buffers, opaque, opaque_len);
//}
void gpu_causal_conv1d_bwd_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_bwd<float, float>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_bwd_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_bwd<__half, __half>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_bwd_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_bwd<__nv_bfloat16, __nv_bfloat16>(stream, buffers, opaque, opaque_len);
}

void gpu_causal_conv1d_bwd_f32_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_bwd<float, __half>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_bwd_f32_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_bwd<float, __nv_bfloat16>(stream, buffers, opaque, opaque_len);
}

void gpu_causal_conv1d_bwd_f16_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_bwd<__half, float>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_bwd_f16_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_bwd<__half, __nv_bfloat16>(stream, buffers, opaque, opaque_len);
}

void gpu_causal_conv1d_bwd_bf16_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_bwd<__nv_bfloat16, float>(stream, buffers, opaque, opaque_len);
}
void gpu_causal_conv1d_bwd_bf16_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len){
    apply_causal_conv1d_bwd<__nv_bfloat16, __half>(stream, buffers, opaque, opaque_len);
}

}  // namespace causal_conv1d_jax