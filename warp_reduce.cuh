#pragma once

template<typename T>
__inline__ __device__
T warpReduceSum(T val) {
  for (int i = warpSize/2; i >= 1; i /= 2)
    val += __shfl_xor_sync(0xffffffff, val, i, 32);
  return val;
}

