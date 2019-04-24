#pragma once

#include "block_reduce.cuh"

template<typename T>
__global__ void device_reduce_stable_kernel(T* in, T* out, int N) {
  T sum=T(0);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    sum+=in[i];
  }
  sum=blockReduceSum(sum);
  if(threadIdx.x==0)
    out[blockIdx.x]=sum;
}

template<typename T>
void device_reduce_stable(T* in, T* out, int N) {
  int threads=512;
  int blocks=min((N+threads-1)/threads,1024);

  device_reduce_stable_kernel<<<blocks,threads>>>(in,out,N);
  device_reduce_stable_kernel<<<1,1024>>>(out,out,blocks); 
}


