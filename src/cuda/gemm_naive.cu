#include <cuda_gemm.hpp>

__global__ void cuda_gemm_naive_kernel(const float * a, const float * b, float *c, int M, int N, int K) {
  unsigned int i = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int j = threadIdx.x + blockIdx.x * blockDim.x;
  if(i >= M || j >= N){
    return;
  }
  float temp_sum = 0.0f;
  #pragma unroll
  for(int k = 0; k < K; k++){
    temp_sum += a[i * K + k] * b[k * N + j];
  }
  c[i * N + j] = temp_sum;
}

void cuda_gemm_naive(const float * a, const float * b, float *c, int M, int N, int K){
  dim3 block(32 , 32);
  dim3 grid((N  - 1) / block.x + 1, (M - 1)/ block.y + 1);
  cuda_gemm_naive_kernel<<< grid, block>>>(a, b, c, M, N, K);
  cudaDeviceSynchronize();
}