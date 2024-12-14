#include <iostream>
#include "kernels.cuh"

#define BLOCK_DIM 1024
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

__global__ void twoSumKernel2(int* data, int data_num, int target, int* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr unsigned int warp_size = 32;
    constexpr unsigned int mask = 0xFFFFFFFF;
    if (idx < data_num)
    {
        int current = data[idx];
        for (int i = idx + 1; i < data_num + warp_size; i+=warp_size)
        {
            int current_data = i < data_num ? data[i] : 0;
            for (int j = 0; j<warp_size; j++)
            {
                int test = __shfl_sync(mask, current_data, j, warp_size);
                if (current + test == target)
                {
                    out[0] = idx;
                    out[1] = i;
                }
            }
        }
    }
}

__global__ void twoSumKernel3(int* data, int data_num, int target, int* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr unsigned int warp_size = 32;
    constexpr unsigned int mask = 0xFFFFFFFF;
    __shared__ int data_shared[BLOCK_DIM];

    if (idx < data_num)
    {
        int current = data[idx];
        for (int i = idx + 1; i < data_num + BLOCK_DIM; i+=BLOCK_DIM)
        {
            int current_data = i < data_num ? data[i] : 0;
            data_shared[threadIdx.x] = current_data;
            for (int j = 0; j<BLOCK_DIM; j+=warp_size)
            {
                // current_data=data_shared[(j+threadIdx.x)%BLOCK_DIM];
                current_data=data_shared[j + (threadIdx.x%warp_size)];
                for (int k = 0; k<warp_size; k++)
                {
                    int test = __shfl_sync(mask, current_data, k, warp_size);
                    if (current + test == target)
                    {
                        out[0] = idx;
                        out[1] = i;
                    }
                }
            }
        }
    }
}

extern "C" void twoSum(int* data, int* out, int target, int data_num, int variant)
{
    int* data_d;
    int* out_d;
    gpuErrchk(cudaMalloc((void**) &data_d, data_num*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &out_d, 2*sizeof(int)));

    gpuErrchk(cudaMemcpy(data_d, data, data_num*sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_size(BLOCK_DIM, 1, 1);
    dim3 grid_size(ceil((float)data_num/block_size.x), 1, 1);

    switch (variant)
    {
        case 1:
            twoSumKernel<<<grid_size, block_size>>>(data_d, data_num, target, out_d);
            break;
        case 2:
            twoSumKernel2<<<grid_size, block_size>>>(data_d, data_num, target, out_d);
            break;
        case 3:
            twoSumKernel3<<<grid_size, block_size>>>(data_d, data_num, target, out_d);
            break;
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(out, out_d, 2*sizeof(int), cudaMemcpyDeviceToHost));
}
