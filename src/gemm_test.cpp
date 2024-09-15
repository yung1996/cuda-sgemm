#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <functional>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cpu_gemm.hpp>
#include <cuda_gemm.hpp>

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

void print_matrix(const std::vector<float>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl; // New line after each row
    }
    std::cout << std::endl;
}

typedef std::function<void(const float*, const float*, float*, int, int, int)> gemm_func;


float run_time_test(gemm_func gemm, int n_iter, const float* a, const float* b, float* c, int M, int N, int K) {

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

float run_time_test_cublas(int n_iter, const float* a, const float* b, float* c, int M, int N, int K) {

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


int main(int argc, char* argv[]) {
  // Check if enough arguments are provided
  if (argc < 5) {
      std::cerr << "Usage: " << argv[0] << " <implementation> <matrix size>" << std::endl;
      std::cerr << "Implementation options: basic, optimized" << std::endl;
      return 1;
  }

  // Determine the GEMM implementation to use
  int implementation = std::stoi(argv[1]);
  // force to align by 4
  int M = (std::stoi(argv[2]) / 4) * 4; // Matrix size M
  int N = (std::stoi(argv[3]) / 4) * 4; // Matrix size
  int K = (std::stoi(argv[4]) / 4) * 4; // Matrix size

  int n_iter = 1;
  if (argc >= 6) {
    n_iter =  std::stoi(argv[5]);
  }

  // Initialize matrices A, B, and C with random values
  std::vector<float> h_A(M * K);
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
  std::uniform_real_distribution<float> dis(0.0, 1.0);

  for (int i = 0; i < M * K; ++i) {
    float random_number = dis(gen);
    h_A[i] = static_cast<float>(random_number);
  }
  for (int i = 0; i < K * N; ++i) {
    float random_number = dis(gen);
    h_B[i] = static_cast<float>(random_number);
  }

  // set up cuda matrix
  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, sizeA));
  CUDA_CHECK(cudaMalloc(&d_B, sizeB));
  CUDA_CHECK(cudaMalloc(&d_C, sizeC));

  // Copy matrices from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), sizeC, cudaMemcpyHostToDevice));

  // Measure time and execute the selected GEMM implementation
  float avg_elapsedTime = 0.0f;
  auto start = std::chrono::high_resolution_clock::now();
  std::string func_name;
  switch (implementation)
  {
    case 0:
      cpu_gemm_naive(h_A.data(), h_B.data(), h_C.data(), M, N, K);
      func_name = "cpu_gemm_naive";
      break;

    case 1:
      avg_elapsedTime = run_time_test(cuda_gemm_naive, n_iter, d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_naive";
      break;
    
    case 2:
      // Perform matrix multiplication: C = alpha * A * B + beta * C
      avg_elapsedTime = run_time_test_cublas(n_iter, d_A, d_B, d_C, M, N, K);
      func_name = "cuda_gemm_cublas";
      break;
    
    case 3:
      avg_elapsedTime = run_time_test(cuda_gemm_float4, n_iter, d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_float4";
      break;

    case 4:
      avg_elapsedTime = run_time_test(cuda_gemm_8x8_float4, n_iter, d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_8x8_float4";
      break;
  
    case 5:
      avg_elapsedTime = run_time_test(cuda_gemm_8x8_float4_2, n_iter, d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_8x8_float4_2";
      break;

    case 6:
      avg_elapsedTime = run_time_test(cuda_gemm_8x8_float4_3, n_iter, d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_8x8_float4_3";
      break;

    case 7:
      avg_elapsedTime = run_time_test(cuda_gemm_smem_float4, n_iter, d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_smem_float4";
      break;

    case 8:
      avg_elapsedTime = run_time_test(cuda_gemm_double_smem_float4, n_iter, d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_double_smem_float4";
      break;

    case 9:
      avg_elapsedTime = run_time_test(cuda_gemm_double_smem_double_float4, n_iter, d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_double_smem_double_float4";
      break;

    case 10:
      avg_elapsedTime = run_time_test(cuda_gemm_double_smem_2x1_float4, n_iter, d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_double_smem_2x1_float4";
      break;

    case 11:
      avg_elapsedTime = run_time_test(cuda_gemm_double_smem_4x1_float4, n_iter, d_A, d_B, d_C, M, N, K);
      // cuda_gemm_double_smem_4x1_float4(d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_double_smem_4x1_float4";
      break;

    default:
      break;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  // copy result back
  if (implementation != 0) {
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
  }

  // Check for kernel launch errors
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy the result matrix back to host
  std::cout << "Test Function for GEMM: " << func_name << std::endl;
  std::cout << "Total Elapsed Time: " << duration.count() << " (second)" << std::endl;
  std::cout << "Average Elapsed Time: " << avg_elapsedTime << "(millisecond)." << std::endl;
  // Free device memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  // std::cout << "***********" << " Matrix A " << "***********" << std::endl;
  // print_matrix(h_A, M, K);
  // std::cout << "***********" << " Matrix B " << "***********" << std::endl;
  // print_matrix(h_B, K, N);
  // std::cout << "***********" << " Matrix C " << "***********" << std::endl;
  // print_matrix(h_C, M, N);
  // std::cout << h_C[0] << ", " <<  h_C[100] << std::endl;
  // cpu_gemm_naive(h_A.data(), h_B.data(), h_C.data(), M, N, K);
  // std::cout << h_C[0] << ", " <<  h_C[100] << std::endl;
  // std::cout << "***********" << " Matrix A " << "***********" << std::endl;
  // print_matrix(h_A, M, K);
  // std::cout << "***********" << " Matrix B " << "***********" << std::endl;
  // print_matrix(h_B, K, N);
  // std::cout << "***********" << " Matrix C " << "***********" << std::endl;
  // print_matrix(h_C, M, N);
  return 0;
}