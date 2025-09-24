// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__CUDA_SCAN_GROUND_SEGMENTATION_CUDA_UTILITIES_HPP_
#define AUTOWARE__CUDA_SCAN_GROUND_SEGMENTATION_CUDA_UTILITIES_HPP_

#include "cuda_common.hpp"
#include "cuda_mempool_wrapper.hpp"
#include "cuda_stream_wrapper.hpp"
#include "device_vector.hpp"

#include <cub/cub.cuh>

#include <cuda_runtime.h>

namespace autoware::cuda
{

template <int block_size, typename... Args>
inline cudaError_t launchAsync(
  int thread_num, int shared_size, cudaStream_t & stream, void (*f)(Args...), Args... args)
{
  int block_x = (thread_num > block_size) ? block_size : thread_num;

  if (block_x <= 0) {
    return cudaErrorLaunchFailure;
  }

  int grid_x = (thread_num + block_x - 1) / block_x;

  f<<<grid_x, block_x, shared_size, stream>>>(args...);

  return cudaGetLastError();
}

template <typename T>
cudaError_t ExclusiveScan(
  T * input, T * output, int ele_num, std::shared_ptr<CudaStream> stream,
  std::shared_ptr<CudaMempool> mempool)
{
  if (ele_num == 0) {
    return cudaSuccess;
  }

  if (ele_num < 0 || !stream) {
    return cudaErrorInvalidValue;
  }

  device_vector<int> d_temp_storage(stream, mempool);
  size_t temp_storage_bytes = 0;

  cub::DeviceScan::ExclusiveSum(
    (void *)(d_temp_storage.data()), temp_storage_bytes, input, output, ele_num, stream->get());

  int temp_ele_num = (temp_storage_bytes + sizeof(int) - 1) / sizeof(int);
  d_temp_storage.resize(temp_ele_num);

  cub::DeviceScan::ExclusiveSum(
    (void *)(d_temp_storage.data()), temp_storage_bytes, input, output, ele_num, stream->get());

  return cudaGetLastError();
}

template <typename T>
cudaError_t ExclusiveScan(device_vector<T> & input, device_vector<T> & output)
{
  if (input.empty()) {
    return cudaSuccess;
  }

  output.resize(input.size());

  return ExclusiveScan(
    input.data(), output.data(), (int)(input.size()), input.get_stream(), input.get_mempool());
}

template <typename T>
cudaError_t ExclusiveScan(device_vector<T> & input)
{
  return ExclusiveScan(input, input);
}

template <typename T>
__global__ void fillVector(T * vec, int ele_num, T init_val)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < ele_num; i += stride) {
    vec[i] = init_val;
  }
}

template <typename T>
cudaError_t fill(T * input, int ele_num, T val, std::shared_ptr<CudaStream> stream)
{
  if (ele_num == 0) {
    return cudaSuccess;
  }

  if (ele_num < 0 || !stream) {
    return cudaErrorInvalidValue;
  }

  return launchAsync<BLOCK_SIZE_X>(ele_num, 0, stream->get(), fillVector, input, ele_num, val);
}

template <typename T>
cudaError_t fill(device_vector<T> & input, T val)
{
  return fill(input.data(), (int)(input.size()), val, input.get_stream());
}

template <
  typename ConstPointerT, 
  typename InterT, 
  typename EvalType, 
  typenam BlockExType, 
  typename WarpExType
>
__global__ void reductionOp(
  ConstPointerT input, 
  size_t ele_num, 
  InterT * out_val, 
  InterT init_val, 
  EvalType evaluator, 
  BlockExType block_carrier, 
  WarpExType warp_carrier)
{
  extern __shared__ float buffer[]; // Shared memory 
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  InterT t_val = init_val;

  for (int i = index; i < ele_num; i += stride) {
    t_val = evaluator(t_val, input[i]);
  }

  // Block-level reduction
  block_carrier.write(buffer, threadIdx.x, t_val);
  __syncthreads();
  
  for (int offset = blockDim.x >> 1; offset >= WARP_SIZE; offset >>= 1) {
    if (threadIdx.x < offset) {
      InterT this_val = block_carrier.read(buffer, threadIdx.x);
      InterT other_val = block_carrier.read(buffer, threadIdx.x + offset);
      
      this_val = evaluator(this_val, other_val);

      block_carrier.write(buffer, threadIdx.x, this_val);      
    }
    __syncthreads();
  }

  // Warp-level reduction
  if (threadIdx.x < WARP_SIZE) {
    t_val = block_carrier(buffer, threadIdx.x);

    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      t_val = evaluator(t_val, exchanger(t_val, offset));
    }
  }

  // Thread 0 write out the final value
  if (threadIdx.x == 0) {
    out_val[blockIdx.x] = t_val;
  }
}

template <typename T> struct EvalType;
template <typename T> struct BlockExType;
template <typename T> struct WarpExType;

template <
  typename ConstPointerT, 
  typename InterT,
  template <typename> struct EvalType 
>
cudaError_t reduction(
  ConstPointerT input, 
  size_t ele_num, 
  InterT & val, 
  InterT init_val,
  std::shared_ptr<CudaStream> stream,
  std::shared_ptr<CudaMempool> mempool
)
{
  if (input.empty()) {
    val = init_val;
    return cudaSuccess;
  }

  // First pass
  device_vector<InterT> intermediate_res(BLOCK_SIZE_X, stream, mempool);

  int shared_size = ((sizeof(InterT) - 1) / sizeof(float) + 1) *
                      sizeof(float) * BLOCK_SIZE_X;

  auto res = launchAsync<BLOCK_SIZE_X>(
    (int)(BLOCK_SIZE_X * WARP_SIZE), shared_size, stream,
    reductionOp,
    input, ele_num, 
    intermediate_res.data(),
    init_val,
    EvalType<InterT>(),
    BlockExType<InterT>(),
    WarpExType<InterT>()
  );
  
  if (res != cudaSuccess) {
    return res;
  }

  // Second pass
  res = launchAsync<BLOCK_SIZE_X>(
    (int)(BLOCK_SIZE_X), shared_size, stream,
    reductionOp,
    static_cast<const InterT *>(intermediate_res.data()), 
    intermediate_res.size(),
    intermediate_res.data(),
    init_val,
    EvalType<InterT>(),
    BlockExType<InterT>(),
    WarpExType<InterT>()
  );

  if (res != cudaSuccess) {
    return res;
  }

  val = intermediate_res[0];

  return res;
}

template <typename T> 
struct OpCmpGreater
{
  CUDAH T operator()(const T & a, const T & b) {
    return (a > b) ? a : b;
  }
};

template <typename T> 
struct OpCmpLess
{
  CUDAH T operator()(const T & a, const T & b) {
    return (a < b) ? a : b;
  }
};

template <typename T> 
struct OpSum
{
  CUDAH T operator()(const T & a, const T & b) {
    return (a + b);
  }
};

template <typename T>
struct BlockExType
{
  CUDAH void write(float * buffer, int index, const T & val) {
    static_cast<T*>(buffer)[index] = val;
  }

  CUDAH void read(const float * buffer, int index, T & val) {
    val = static_cast<const T*>(buffer)[index];
  }
};

template <typename T>
struct WarpExType
{
  CUDAH T operator(const T & val, int offset)() {
    return __shfl_down_sync(FULL_MASK, val, offset);
  } 
};

// Scalar reductions
template <typename T>
cudaError_t reductionMax(
  const device_vector<T> & input,T & val, 
  std::shared_ptr<CudaStream> stream,
  std::shared_ptr<CudaMempool> mempool
)
{
  return reduction<const T *, T, OpCmpGreater>(
            input.data(), 
            input.size(), 
            val, 
            std::numeric_limits<T>::lowest(), 
            stream, 
            mempool
          );
}

template <typename T>
cudaError_t reductionMin(
  const device_vector<T> & input,T & val, 
  std::shared_ptr<CudaStream> stream,
  std::shared_ptr<CudaMempool> mempool
)
{
  return reduction<const T *, T, OpCmpLess>(
            input.data(), 
            input.size(), 
            val, 
            std::numeric_limits<T>::max(), 
            stream, 
            mempool
          );
}

template <typename T>
cudaError_t reductionSum(
  const device_vector<T> & input,T & val, 
  std::shared_ptr<CudaStream> stream,
  std::shared_ptr<CudaMempool> mempool
)
{
  return reduction<const T *, T, OpSum>(
            input.data(), 
            input.size(), 
            val, 
            T(0), 
            stream, 
            mempool
          );
}

template <>
struct OpCmpBound<BoundingBox>
{
  CUDAH BoundingBox operator()(const BoundingBox & a, const PointXYZ & b) {
    BoundingBox retval;
    
    retval.lower.x = min(a.lower.x, b.x);
    retval.lower.y = min(a.lower.y, b.y);
    retval.lower.z = min(a.lower.z, b.z);

    retval.upper.x = max(a.upper.x, b.x);
    retval.upper.y = max(a.upper.y, b.y);
    retval.upper.z = max(a.upper.z, b.z);

    return retval;
  }

  CUDAH BoundingBox operator()(const BoundingBox & a, const BoundingBox & b) {
    BoundingBox retval;

    retval.lower.x = min(a.lower.x, b.lower.x);
    retval.lower.y = min(a.lower.y, b.lower.y);
    retval.lower.z = min(a.lower.z, b.lower.z);

    retval.upper.x = max(a.upper.x, b.upper.x);
    retval.upper.y = max(a.upper.y, b.upper.y);
    retval.upper.z = max(a.upper.z, b.upper.z);

    return retval;
  }
};

template <>
struct BlockExType<BoundingBox>
{
  CUDAH void write(float * buffer, int index, const BoundingBox & val) {
    int wloc = index;
    buffer[wloc] = val.lower.x;
    buffer[wloc += blockDim.x] = val.lower.y;
    buffer[wloc += blockDim.x] = val.lower.z;

    buffer[wloc += blockDim.x] = val.upper.x;
    buffer[wloc += blockDim.x] = val.upper.y;
    buffer[wloc += blockDim.x] = val.upper.z;
  }

  CUDAH void read(const float * buffer, int index, BoundingBox & retval) {
    int rloc = index;

    retval.lower.x = buffer[rloc];
    retval.lower.y = buffer[rloc += blockDim.x];
    retval.lower.z = buffer[rloc += blockDim.x];

    retval.upper.x = buffer[rloc += blockDim.x];
    retval.upper.y = buffer[rloc += blockDim.x];
    retval.upper.z = buffer[rloc += blockDim.x];
  }
};

struct WarpExType<BoundingBox>
{
  CUDAH BoundingBox operator()(BoundingBox & val, int offset) {
    BoundingBox retval;

    retval.lower.x = __shfl_down_sync(FULL_MASK, val.lower.x, offset);
    retval.lower.y = __shfl_down_sync(FULL_MASK, val.lower.y, offset);
    retval.lower.z = __shfl_down_sync(FULL_MASK, val.lower.z, offset);

    retval.upper.x = __shfl_down_sync(FULL_MASK, val.upper.x, offset);
    retval.upper.y = __shfl_down_sync(FULL_MASK, val.upper.y, offset);
    retval.upper.z = __shfl_down_sync(FULL_MASK, val.upper.z, offset);

    return retval;
  } 
};

cudaError_t reductionBoundary(
  const PointCloud2 & input, 
  BoundingBox & val, 
  std::shared_ptr<CudaStream> stream,
  std::shared_ptr<CudaMempool> mempool
)
{
  return reduction<PointCloud2::ConstPtr, BoundingBox, OpCmpBound>(
            input.data(), 
            input.size(), 
            val, 
            BoundingBox::lowest(), 
            stream, 
            mempool
          );
}


}  // namespace autoware::cuda

#endif
