#include <cpu_gemm.hpp>
#include <iostream>

void cpu_gemm_naive(const float * a, const float * b, float * c, int M, int N, int K) {

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {

      float temp_sum = 0.0;

      for (int k = 0; k < K; ++k) {
        temp_sum += a[i * K + k] * b[k * N + j];
      }
      c[i * N + j] = temp_sum;
    }
  }
}