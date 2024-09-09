
__global__ void cuda_gemm_naive_kernel(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_naive(const float * a, const float * b, float * c, int M, int N, int K);

void cuda_gemm_float4(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_8x8_float4(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_8x8_float4_2(const float * a, const float * b, float *c, int M, int N, int K);

void cuda_gemm_8x8_float4_3(const float * a, const float * b, float *c, int M, int N, int K);