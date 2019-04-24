#pragma once

#include "warp_reduce.cuh"

template<typename T>
__inline__ __device__
T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane=threadIdx.x%warpSize;
  int wid=threadIdx.x/warpSize;
  val=warpReduceSum(val);

  //write reduced value to shared memory
  if(lane==0) shared[wid]=val;
  __syncthreads();

  //ensure we only grab a value from shared memory if that warp existed
  val = (threadIdx.x<blockDim.x/warpSize) ? shared[lane] : int(0);
  if(wid==0) val=warpReduceSum(val);

  return val;
}
