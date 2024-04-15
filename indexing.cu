#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void sum(float *k)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = blockIdx.x;
    int local_tid = threadIdx.x;
    printf("global_tid:%d block_id:%d local_tid:%d\n", global_tid, block_id, local_tid);
    k[global_tid] = k[global_tid] + 1;
}
int main()
{
    int N = 32;
    int size = N * sizeof(float);
    float *hx;
    float *dx;
    hx = (float *)malloc(size);
    cudaMalloc((void **)&dx, size);
    for (int i = 0; i < N; i++)
    {
        hx[i] = i;
    }
    cudaMemcpy(dx, hx, size, cudaMemcpyHostToDevice);
    sum<<<1, 32>>>(dx);
    cudaMemcpy(hx, dx, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
    {
        printf("%f\n", hx[i]);
    }
    free(hx);
    cudaFree(dx);
}