/*************************************************************************
  > File Name   : threadfence.cpp
  > Author      : Liu Junhong
  > Mail        : junliu@nvidia.com
  > Created Time: Tuesday, February 11, 2020 PM11:44:49 HKT
 ************************************************************************/

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

__device__ unsigned int count = 0;
__shared__ bool isLastBlockDone;

__device__ float calculatePartialSum(const float* array, int N) {

    __shared__ float tmp[32];
    const float* local_array = array + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int laneId = threadIdx.x & 31;
    int warpId = threadIdx.x / 32;
    int warpNum = blockDim.x / 32;
    float sum = 0;
    if (threadIdx.x + blockIdx.x * blockDim.x < N) {
        sum = local_array[tid];
    }
    for (int i = 16; i >= 1; i /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, i, 32);
    }

    if(laneId == 0) {
        tmp[warpId] = sum;
    }
    __syncthreads();
    if(warpId == 0) {
        if (laneId < warpNum) {
            sum = tmp[laneId];
        }
        else {
            sum = 0;
        }

        for (int i = 16; i >= 1; i /= 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, i, 32);
        }
    }

    __syncthreads();

    return sum;
}
 
__device__ float calculateTotalSum(volatile float* result) {

    int tid = threadIdx.x;
    int laneId = threadIdx.x & 31;
    int warpId = threadIdx.x / 32;
    int warpNum = blockDim.x / 32;
    float sum = 0;

    if (gridDim.x < 32) {
        if (warpId == 0) {
            if (laneId < gridDim.x) {
                sum = result[laneId];
            }
            else {
                sum = 0;
            }
            for (int i = 16; i >= 1; i /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, i, 32);
            }
        }
    }
    else {
        for (int i = tid; i < gridDim.x; i+= blockDim.x) {
            sum += result[i];
        }

        __shared__ float tmp[32];

        for (int i = 16; i >= 1; i /= 2) {
            sum += __shfl_xor_sync(0xffffffff, sum, i, 32);
        }
        if(laneId == 0) {
            tmp[warpId] = sum;
        }
        __syncthreads();
        if(warpId == 0) {
            if (laneId < warpNum) {
                sum = tmp[laneId];
            }
            else {
                sum = 0;
            }

            for (int i = 16; i >= 1; i /= 2) {
                sum += __shfl_xor_sync(0xffffffff, sum, i, 32);
            }
        }

        __syncthreads();
    }

    return sum;

}

__global__ void sum(const float* array, unsigned int N, volatile float* result) {

    float partialSum = calculatePartialSum(array, N);
    if (threadIdx.x == 0) {
        result[blockIdx.x] = partialSum;
        __threadfence();  // attention
        unsigned int value = atomicInc(&count, gridDim.x);
        isLastBlockDone = (value == (gridDim.x - 1));
    }

    __syncthreads(); //must have 

    if (isLastBlockDone) {
        float totalSum = calculateTotalSum(result);
        if (threadIdx.x == 0) {
            result[0] = totalSum;
            count = 0;
        }
    }
}

int main(int argc, char **argv){


    int N = 1048;
    float* cpu_array = (float*)malloc(sizeof(float) * N);
    float* gpu_array;
    float* gpu_result;
    cudaMalloc((void**)&gpu_array, sizeof(float)*N);

    for(int i = 0; i < N; i++) {
        cpu_array[i] = i;
    }


    cudaMemcpy(gpu_array, cpu_array, sizeof(float)*N, cudaMemcpyHostToDevice);

    int threadsNum = 256;
    int blocksNum = (N + threadsNum - 1) / threadsNum;

    cudaMalloc((void**)&gpu_result, sizeof(float)*blocksNum);
    float* cpu_result = (float*)malloc(sizeof(float)*blocksNum);

    sum<<<blocksNum, threadsNum>>>(gpu_array, N, gpu_result);

    cudaMemcpy(cpu_result, gpu_result, sizeof(float)*blocksNum, cudaMemcpyDeviceToHost);
//        for (int i = 0; i < blocksNum; i++) {
//            printf("%f\t", cpu_result[i]);
//        }
//        printf("\n");

    printf("%f \n", cpu_result[0]);

    return 0;
}
