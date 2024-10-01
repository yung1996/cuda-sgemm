#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <utility>
#include <tuple>
#include <functional>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cpu_gemm.hpp>
#include <cuda_gemm.hpp>
#include <sgemm.hpp>

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

typedef std::function<void(const float* a, const float* b, float* c, int M, int N, int K)> gemmFunction;

struct TestCase {
  size_t M, N, K;
};

void transpose(const float* input, float* output, int rows, int cols) {
    // Loop over the input matrix
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Calculate the corresponding index in the 1D array
            // Transpose element from (i, j) to (j, i)
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

float run_single_cublas(int n_iter, const float* a, const float* b, float* c, int M, int N, int K) {

  // Create cuBLAS handle
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  float alpha = 1.0f;
  float beta = 0.0f;


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float total_elapsedTime = 0.0f;
  // warm up
  CUBLAS_CHECK(cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, a, K, b, N, &beta, c, M));
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int i = 0; i < n_iter; ++i) {
    float elapsedTime = 0.0f;

    cudaEventRecord(start, 0);

    cublasSgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, a, K, b, N, &beta, c, M);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    total_elapsedTime += elapsedTime;

    // synchronization
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return total_elapsedTime / n_iter;

}


float run_single_test(gemmFunction gemm, int n_iter, const float* a, const float* b, float* c, int M, int N, int K) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float total_elapsedTime = 0.0f;
  // warm up
  warmup(a, b, c, M, N, K);

  for (int i = 0; i < n_iter; ++i) {
    float elapsedTime = 0.0f;

    cudaEventRecord(start, 0);

    gemm(a, b, c, M, N, K);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    total_elapsedTime += elapsedTime;

    // synchronization
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return total_elapsedTime / n_iter;
}

void run_tests(std::vector<std::tuple<std::string, gemmFunction, bool>>& functions, std::vector<TestCase>& test_cases, int n_iters) {
  for (size_t i = 0; i < functions.size(); ++i) {
    std::string func_name = std::get<0>(functions[i]);
    gemmFunction gemm_func = std::get<1>(functions[i]);
    bool if_transpose = std::get<2>(functions[i]);
    std::cout << "********************************************************" << std::endl;
    std::cout << "Function Name: " << func_name << std::endl;

    for (const auto& test : test_cases) {
      // basic configuration
      size_t M = test.M;
      size_t N = test.N;
      size_t K = test.K;

      // Initialize matrices A, B, and C with random values
      std::vector<float> h_A(M * K);
      std::vector<float> h_AT(K * M); // used for transpose version
      std::vector<float> h_B(K * N);
      std::vector<float> h_C(M * N, 0.0f);

      // setup cublas
      // Create cuBLAS handle
      cublasHandle_t handle;
      CUBLAS_CHECK(cublasCreate(&handle));
      float alpha = 1.0f;
      float beta = 0.0f;

      // Fill matrices A and B with random values
      std::random_device rd;  // Non-deterministic random number generator
      std::mt19937 gen(rd()); // Seed the generator

      // Define the range [0, 1.0]
      std::uniform_real_distribution<float> dis(0.0, 0.1);

      for (int i = 0; i < M * K; ++i) {
        float random_number = dis(gen);
        h_A[i] = static_cast<float>(random_number);
      }
      for (int i = 0; i < K * N; ++i) {
        float random_number = dis(gen);
        h_B[i] = static_cast<float>(random_number);
      }

      // row major
      transpose(h_A.data(), h_AT.data(), M, K);

      // set up cuda matrix
      size_t sizeA = M * K * sizeof(float);
      size_t sizeB = K * N * sizeof(float);
      size_t sizeC = M * N * sizeof(float);

      float *d_A, *d_B, *d_C, *d_AT;
      CUDA_CHECK(cudaMalloc(&d_A, sizeA));
      CUDA_CHECK(cudaMalloc(&d_AT, sizeA));
      CUDA_CHECK(cudaMalloc(&d_B, sizeB));
      CUDA_CHECK(cudaMalloc(&d_C, sizeC));

      // Copy matrices from host to device
      CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_AT, h_AT.data(), sizeA, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), sizeC, cudaMemcpyHostToDevice));

      // run the gemm functions
      float avg_elapsedTime = if_transpose ? \
        run_single_test(gemm_func, n_iters, d_AT, d_B, d_C, M, N, K) : \
        run_single_test(gemm_func, n_iters, d_A, d_B, d_C, M, N, K);

      // Check for kernel launch errors
      CUDA_CHECK(cudaPeekAtLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      // Copy the result matrix back to host
      std::cout << "Matrix Setting, " << "M: "<< M << "N: " << N << "K: " << K << "; ";
      std::cout << "Average Elapsed Time: " << avg_elapsedTime << "(millisecond); ";
      // compute tflops
      long double flops = 2LL * M * N * K;
      long double tflops = (flops / (avg_elapsedTime * 1e-3)) * 1e-12;
      std::cout << "TFlops: " << tflops << std::endl;
      // Free device memory
      CUDA_CHECK(cudaFree(d_A));
      CUDA_CHECK(cudaFree(d_AT));
      CUDA_CHECK(cudaFree(d_B));
      CUDA_CHECK(cudaFree(d_C));
    }
  }
  std::cout << "********************************************************" << std::endl;
  std::cout << "Cublas" << std::endl;

  // calculate the cublas usage
  for (const auto& test : test_cases) {
    // basic configuration
    size_t M = test.M;
    size_t N = test.N;
    size_t K = test.K;

    // Initialize matrices A, B, and C with random values
    std::vector<float> h_A(M * K);
    std::vector<float> h_AT(K * M); // used for transpose version
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);

    // setup cublas
    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f;
    float beta = 0.0f;

    // Fill matrices A and B with random values
    std::random_device rd;  // Non-deterministic random number generator
    std::mt19937 gen(rd()); // Seed the generator

    // Define the range [0, 1.0]
    std::uniform_real_distribution<float> dis(0.0, 0.1);

    for (int i = 0; i < M * K; ++i) {
      float random_number = dis(gen);
      h_A[i] = static_cast<float>(random_number);
    }
    for (int i = 0; i < K * N; ++i) {
      float random_number = dis(gen);
      h_B[i] = static_cast<float>(random_number);
    }

    // row major
    transpose(h_A.data(), h_AT.data(), M, K);

    // set up cuda matrix
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *d_A, *d_B, *d_C, *d_AT;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_AT, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    // Copy matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_AT, h_AT.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), sizeC, cudaMemcpyHostToDevice));

    // Measure time and execute the selected GEMM implementation
    

    // run the gemm functions
    float avg_elapsedTime = run_single_cublas(n_iters, d_A, d_B, d_C, M, N, K);

    // Check for kernel launch errors
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result matrix back to host
    std::cout << "Matrix Setting, " << "M: "<< M << "N: " << N << "K: " << K << "; ";
    std::cout << "Average Elapsed Time: " << avg_elapsedTime << "(millisecond); ";
    // compute tflops
    long double flops = 2LL * M * N * K;
    long double tflops = (flops / (avg_elapsedTime * 1e-3)) * 1e-12;
    std::cout << "TFlops: " << tflops << std::endl;
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_AT));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
  }
}


int main() {
  // iterations number
  int n_iters = 100;
  // Define test cases
  std::vector<TestCase> test_cases = {
    {128, 128, 128},
    {256, 256, 256},
    {512, 512, 512},
    {1024, 1024, 1024},
    {2048, 2048, 2048},
    {4096, 4096, 4096}
    // {8192, 8192, 8192},
    // {16384, 16384, 16384}
  };
  // Define test gemm functions
  std::vector<std::tuple<std::string, gemmFunction, bool>> gemmFuncNamePairs {
    {"cuda_gemm_naive", cuda_gemm_naive, false},
    {"cuda_gemm_float4", cuda_gemm_float4, false},
    {"cuda_gemm_8x8_float4", cuda_gemm_8x8_float4, false},
    {"cuda_gemm_8x8_float4_2", cuda_gemm_8x8_float4_2, false},
    {"cuda_gemm_8x8_float4_3", cuda_gemm_8x8_float4_3, false},
    {"cuda_gemm_smem_float4", cuda_gemm_smem_float4, false},
    {"cuda_gemm_double_smem_float4", cuda_gemm_double_smem_float4, false},
    {"cuda_gemm_double_smem_2x1_float4", cuda_gemm_double_smem_2x1_float4, false},
    {"cuda_gemm_double_smem_4x1_float4", cuda_gemm_double_smem_4x1_float4, false},
    {"sgemm_128x128x8", sgemm_128x128x8, true}
  };

    // Run the tests
    run_tests(gemmFuncNamePairs, test_cases, n_iters);

    return 0;
}