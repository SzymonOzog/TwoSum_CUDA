#include <iostream>
#include "kernels.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void twoSumKernel(float* nums, int target, int* out)
{
    out[0] = 0;
    out[1] = 1;
}

extern "C" void twoSum(float* data, int* out, int target, int data_num)
{
    float* data_d;
    int* out_d;
    gpuErrchk(cudaMalloc((void**) &data_d, data_num*sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &out_d, 2*sizeof(int)));

    gpuErrchk(cudaMemcpy(data_d, data, data_num*sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid_size(1, 1, 1);
    dim3 block_size(1, 1, 1);

    twoSumKernel<<<grid_size, block_size>>>(data_d, target, out_d);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(out, out_d, 2*sizeof(int), cudaMemcpyDeviceToHost));
}
