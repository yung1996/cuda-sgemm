#include <cuda_gemm.hpp>
#include <stdio.h>

inline __device__ void mma4_4(const float4 fragmentA, const float4 fragmentB, float4 * tc){
  tc[0].x += fragmentA.x * fragmentB.x;
  tc[0].y += fragmentA.x * fragmentB.y;
  tc[0].z += fragmentA.x * fragmentB.z;
  tc[0].w += fragmentA.x * fragmentB.w;

  tc[1].x += fragmentA.y * fragmentB.x;
  tc[1].y += fragmentA.y * fragmentB.y;
  tc[1].z += fragmentA.y * fragmentB.z;
  tc[1].w += fragmentA.y * fragmentB.w;

  tc[2].x += fragmentA.z * fragmentB.x;
  tc[2].y += fragmentA.z * fragmentB.y;
  tc[2].z += fragmentA.z * fragmentB.z;
  tc[2].w += fragmentA.z * fragmentB.w;

  tc[3].x += fragmentA.w * fragmentB.x;
  tc[3].y += fragmentA.w * fragmentB.y;
  tc[3].z += fragmentA.w * fragmentB.z;
  tc[3].w += fragmentA.w * fragmentB.w;
}

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

__global__ void cuda_gemm_float4_kernel(const float * a, const float * b, float *c, int M, int N, int K) {
  unsigned int i = (threadIdx.y + blockIdx.y * blockDim.y) * 4;
  unsigned int j = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
  if(i >= M || j >= N){
    return;
  }

  // boundary case, assume the memory is align by four
  // if (M - i < 4 || N - j < 4) {
  //   for (int ii = i; ii < M; ++ii) {
  //     for (int jj = j; jj < N; ++jj) {
  //       float temp_sum = 0.0;
  //       for (int k = 0; k < K; ++k) {
  //         temp_sum += a[ii * K + k] * b[k * N + jj];
  //       }
  //       c[ii * N + jj] = temp_sum;
  //     }
  //   }
  //   return;
  // }

  float4 tc_zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 tc[4] = {tc_zero,tc_zero,tc_zero,tc_zero};

  #pragma unroll
  for(int k = 0; k < K; k++){
    float4 fragmentA = make_float4(*(a + i * K + k), *(a + (i + 1) * K + k),*(a + (i + 2) * K + k), *(a + (i + 3) * K + k));

    float4 fragmentB = make_float4(*(b + k * N + j), *(b + k * N + j + 1),  *(b + k * N + j + 2),  *(b + k * N + j + 3));
    mma4_4(fragmentA, fragmentB, tc);
  }
  
  #pragma unroll
  for(int ii = 0; ii < 4; ii++){
    float4 * f4c_row = reinterpret_cast<float4 *>(c + (i + ii) * N);
    f4c_row[j / 4] = tc[ii];
  }
}

void cuda_gemm_float4(const float * a, const float * b, float *c, int M, int N, int K) {
  dim3 block(32 , 32);

  int tN = (N - 1) / 4 + 1;
  int tM = (M - 1) / 4 + 1;

  dim3 grid((tN  - 1) / block.x + 1, (tM - 1)/ block.y + 1);
  cuda_gemm_float4_kernel<<< grid, block>>>(a, b, c, M, N, K);
  cudaDeviceSynchronize();
}

__global__ void cuda_gemm_8x8_float4_kernel(const float * a, const float * b, float *c, int M, int N, int K) {
unsigned int i = (threadIdx.y + blockIdx.y * blockDim.y) * 8;
  unsigned int j = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
  if (i >= M || j >= N) {
    return;
  }

  float4 tc_zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 tc[4] = {tc_zero,tc_zero,tc_zero,tc_zero};
  float4 tc4x4[4][4] = {tc[4], tc[4], tc[4], tc[4]};
  float4 * f4c = reinterpret_cast<float4 *>(c);

  #pragma unroll
  for (unsigned int ii = 0; ii < 8; ii += 4) {
    #pragma unroll
    for (unsigned int jj = 0; jj < 8; jj += 4) {
      unsigned int index = ((ii / 4 )* 2 + jj / 4);
      #pragma unroll
      for(int k = 0; k < K; k++){
        float4 fragmentA = make_float4(*(a + (i + ii) * K + k), *(a + (i + ii + 1) * K + k),*(a + (i + ii + 2) * K + k), *(a + (i + ii + 3) * K + k));
        float4 fragmentB = make_float4(*(b + k * N + j + jj), *(b + k * N + j + jj + 1),  *(b + k * N + j + jj + 2),  *(b + k * N + j + jj + 3));
        mma4_4(fragmentA, fragmentB, tc4x4[index]);
      }

      #pragma unroll
      for(int r = 0; r < 4; r++){
        f4c[(i + ii + r) * (N / 4) + ((j + jj) / 4)] = tc4x4[index][r];
      }
    }
  }

}

void cuda_gemm_8x8_float4(const float * a, const float * b, float *c, int M, int N, int K) {
  dim3 block(32 , 32);

  int tN = (N - 1) / 4 + 1;
  int tM = (M - 1) / 4 + 1;

  dim3 grid((tN  - 1) / block.x + 1, (tM - 1)/ block.y + 1);
  cuda_gemm_8x8_float4_kernel<<<grid, block>>>(a, b, c, M, N, K);
  cudaDeviceSynchronize();
}