#include <cstdio>
#include "device_reduce_atomic.cuh"
#include "device_reduce_block_atomic.cuh"
#include "device_reduce_warp_atomic.cuh"
#include "device_reduce_stable.cuh"
#include "common.h"

//typedef float VALUE_TYPE

#define cudaCheckError() {                                          \
  cudaError_t e=cudaGetLastError();                                  \
  if(e!=cudaSuccess) {                                               \
  printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
  exit(0); \
  }                                                                  \
}

void RunTest(char* label, void (*fptr)(VALUE_TYPE* in, VALUE_TYPE* out, int N), int N, int REPEAT, VALUE_TYPE* src, VALUE_TYPE checksum) {
  VALUE_TYPE *in, *out;
  
  //allocate a buffer that is at least large enough that we can ensure it doesn't just sit in l2.
  int MIN_SIZE=4*1024*1024;
  int size=max(int(sizeof(VALUE_TYPE)*N),MIN_SIZE);
  
  //compute mod base for picking the correct buffer
  int mod=size/(N*sizeof(VALUE_TYPE));
  cudaEvent_t start,stop;
  cudaMalloc(&in,size);
  cudaMalloc(&out,sizeof(VALUE_TYPE)*1024);  //only stable version needs multiple elements, all others only need 1
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaCheckError();

  cudaMemcpy(in,src,N*sizeof(VALUE_TYPE),cudaMemcpyHostToDevice);
  
  //warm up
  fptr(in,out,N);
  cudaMemset(out, 0, 1024*sizeof(VALUE_TYPE));

  cudaDeviceSynchronize();
  cudaCheckError();
  cudaEventRecord(start);

  for(int i=0;i<REPEAT;i++) {
    //iterate through different buffers
    int o=i%mod;
    fptr(in+o*N,out,N);
  }
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaCheckError();

  float time_ms;
  cudaEventElapsedTime(&time_ms,start,stop);
  float time_s=time_ms/(float)1e3;

  float GB=(float)N*sizeof(int)*REPEAT;
  float GBs=GB/time_s/(float)1e9;

  VALUE_TYPE sum;
  cudaMemcpy(&sum,out,sizeof(VALUE_TYPE),cudaMemcpyDeviceToHost);
  cudaCheckError();

  char *valid;
  if(sum==checksum) 
    valid="CORRECT";
  else
  {
    valid="INCORRECT";
    printf("sum = %f, checksum = %f\n", sum, checksum);
  }

  printf("%s: %s, Time: %f s, GB/s: %f\n", label, valid, time_s, GBs); 
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(in);
  cudaFree(out);
  cudaCheckError();
}


int main(int argc, char** argv)
{
  if(argc!=3) {
    printf("Usage: ./reduce num_elems repeat\n");
    exit(0);
  }
  int NUM_ELEMS=atoi(argv[1]);
  int REPEAT=atoi(argv[2]);

  printf("NUM_ELEMS: %d, REPEAT: %d\n", NUM_ELEMS, REPEAT);

  VALUE_TYPE* vals=(VALUE_TYPE*)malloc(NUM_ELEMS*sizeof(VALUE_TYPE));
  VALUE_TYPE checksum =0;
  for(int i=0;i<NUM_ELEMS;i++) {
    vals[i]=rand()%4;
    checksum+=vals[i];
//    printf("vals[%d] = %f\n", i, vals[i]);
  }


  RunTest("device_reduce_atomic", device_reduce_atomic,NUM_ELEMS,REPEAT,vals,checksum);
  
  RunTest("device_reduce_warp_atomic",device_reduce_warp_atomic,NUM_ELEMS,REPEAT,vals,checksum);
  
  RunTest("device_reduce_block_atomic",device_reduce_block_atomic,NUM_ELEMS,REPEAT,vals,checksum);
  
  RunTest("device_reduce_stable",device_reduce_stable,NUM_ELEMS,REPEAT,vals,checksum);

  
  free(vals);

}
