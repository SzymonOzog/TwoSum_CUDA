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

__global__ void twoSumKernel(int* data, int data_num, int target, int* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data_num)
    {
        int current = data[idx];
        for (int i = idx + 1; i < data_num; i++)
        {
            if (current + data[i] == target)
            {
                out[0] = idx;
                out[1] = i;
            }
        }
    }
}

extern "C" void twoSum(int* data, int* out, int target, int data_num)
{
    int* data_d;
    int* out_d;
    gpuErrchk(cudaMalloc((void**) &data_d, data_num*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &out_d, 2*sizeof(int)));

    gpuErrchk(cudaMemcpy(data_d, data, data_num*sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_size(1024, 1, 1);
    dim3 grid_size(ceil((float)data_num/block_size.x), 1, 1);

    twoSumKernel<<<grid_size, block_size>>>(data_d, data_num, target, out_d);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(out, out_d, 2*sizeof(int), cudaMemcpyDeviceToHost));
}
