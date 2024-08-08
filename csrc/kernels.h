#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace causal_conv1d_jax {
struct KeplerDescriptor {
  std::int64_t size;
};

enum ElementType { BF16, F16, F32, F64 };

struct CausalConv1dDescriptor {
  using index_t = uint32_t;

  int batch, dim, seqlen, width;

  index_t x_batch_stride;
  index_t x_c_stride;
  index_t x_l_stride;
  index_t weight_c_stride;
  index_t weight_width_stride;
  index_t out_batch_stride;
  index_t out_c_stride;
  index_t out_l_stride;

  //ElementType input_t;
  //ElementType weight_t;

  bool silu_activation=0;

  index_t conv_state_batch_stride;
  index_t conv_state_c_stride;
  index_t conv_state_l_stride;
};

// === Forward functions ===
void gpu_causal_conv1d_fwd_f64(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_causal_conv1d_fwd_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_causal_conv1d_fwd_f16(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_causal_conv1d_fwd_bf16(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

void gpu_causal_conv1d_fwd_f32_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);
void gpu_causal_conv1d_fwd_f32_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);

void gpu_causal_conv1d_fwd_f16_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);
void gpu_causal_conv1d_fwd_f16_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);

void gpu_causal_conv1d_fwd_bf16_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);
void gpu_causal_conv1d_fwd_bf16_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);

// === Backward function ===
void gpu_causal_conv1d_bwd_f64(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_causal_conv1d_bwd_f32(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_causal_conv1d_bwd_f16(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);
void gpu_causal_conv1d_bwd_bf16(cudaStream_t stream, void** buffers, const char* opaque,
                    std::size_t opaque_len);

void gpu_causal_conv1d_bwd_f32_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);
void gpu_causal_conv1d_bwd_f32_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);

void gpu_causal_conv1d_bwd_f16_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);
void gpu_causal_conv1d_bwd_f16_bf16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);

void gpu_causal_conv1d_bwd_bf16_f32(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);
void gpu_causal_conv1d_bwd_bf16_f16(cudaStream_t stream, void** buffers, const char* opaque,
  std::size_t opaque_len);
}  // namespace causal_conv1d_jax