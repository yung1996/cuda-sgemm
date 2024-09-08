#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <random>

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

int main(int argc, char* argv[]) {
  // Check if enough arguments are provided
  if (argc < 5) {
      std::cerr << "Usage: " << argv[0] << " <implementation> <matrix size>" << std::endl;
      std::cerr << "Implementation options: basic, optimized" << std::endl;
      return 1;
  }

  // Determine the GEMM implementation to use
  int implementation = std::stoi(argv[1]);
  int M = std::stoi(argv[2]); // Matrix size M
  int N = std::stoi(argv[3]); // Matrix size
  int K = std::stoi(argv[4]); // Matrix size

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

  // Define the range [0, 10]
  std::uniform_int_distribution<int> dis(0, 10);

  for (int i = 0; i < M * K; ++i) {
    int random_number = dis(gen);
    h_A[i] = static_cast<float>(random_number);
  }
  for (int i = 0; i < K * N; ++i) {
    int random_number = dis(gen);
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
  auto start = std::chrono::high_resolution_clock::now();
  std::string func_name;
  switch (implementation)
  {
    case 0:
      cpu_gemm_naive(h_A.data(), h_B.data(), h_C.data(), M, N, K);
      func_name = "cpu_gemm_naive";
      break;

    case 1:
      cuda_gemm_naive(d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_naive";
      break;
    
    case 2:
      // Perform matrix multiplication: C = alpha * A * B + beta * C
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
      func_name = "cuda_gemm_cublas";
      break;
    
    case 3:
      cuda_gemm_float4(d_A, d_B, d_C, M, N, K);
      CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      func_name = "cuda_gemm_float4";
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
  std::cout << "Time for GEMM (" << func_name << "): " 
            << duration.count() << " seconds" << std::endl;

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

  // cpu_gemm_naive(h_A.data(), h_B.data(), h_C.data(), M, N, K);
  // std::cout << "***********" << " Matrix A " << "***********" << std::endl;
  // print_matrix(h_A, M, K);
  // std::cout << "***********" << " Matrix B " << "***********" << std::endl;
  // print_matrix(h_B, K, N);
  // std::cout << "***********" << " Matrix C " << "***********" << std::endl;
  // print_matrix(h_C, M, N);
  return 0;
}