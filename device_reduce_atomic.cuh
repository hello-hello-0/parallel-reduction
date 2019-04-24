#pragma once


template<typename T>
__global__ void device_reduce_atomic_kernel(T* in, T* out, int N) {
  T sum=T(0);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    sum+=in[i];
  }
  atomicAdd(out,sum);
}

template<typename T>
void device_reduce_atomic(T* in, T* out, int N) {
  int threads=256;
  int blocks=min((N+threads-1)/threads,2048);

  cudaMemsetAsync(out,0,sizeof(int));
  device_reduce_atomic_kernel<<<blocks,threads>>>(in,out,N); 
}


