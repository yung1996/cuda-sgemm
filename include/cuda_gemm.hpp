
__global__ void cuda_gemm_naive_kernel(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_naive(const float * a, const float * b, float * c, int M, int N, int K);

void cuda_gemm_float4(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_8x8_float4(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_8x8_float4_2(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_8x8_float4_3(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_smem_float4(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_double_smem_float4(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_double_smem_double_float4(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_double_smem_2x1_float4(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_double_smem_4x1_float4(const float * a, const float * b, float *c, int M, int N, int K);

void warmup(const float * a, const float * b, float *c, int M, int N, int K);