#define M_TILE 128
#define N_TILE 128
#define K_TILE 8

#include <sgemm.hpp>

inline __device__ __host__ u_int32_t div_ceil(u_int32_t a, u_int32_t b) {
    return (a - 1) / b + 1;
}

// __constant__ u_int32_t M_TILE = 128;
// __constant__ u_int32_t N_TILE = 128;
// __constant__ u_int32_t K_TILE = 8;

__constant__ u_int32_t FLOAT4_WORD_NUM = sizeof(float4) / sizeof(float);

// Warp id stride
// __constant__ u_int32_t warp_x_stride = 64 / FLOAT4_WORD_NUM;
// __constant__ u_int32_t warp_y_stride = 2 * warp_x_stride;

// Lane id Stride
// __constant__ u_int32_t lane_x_stride = 1;
// __constant__ u_int32_t lane_y_stride = 8;

__global__ void sgemm_128x128x8_kernel(const float * a, const float * b, float *c, int M, int N, int K) {
  // shared memory
  __shared__ __align__(32 *sizeof(float)) float smem[2 * M_TILE * K_TILE + 2 * K_TILE * N_TILE];

  float4 * smem_a = reinterpret_cast<float4*>(smem);
  float4 * smem_b = reinterpret_cast<float4*>(smem + 2 * M_TILE * K_TILE);

  float4 * glb_a = reinterpret_cast<float4*>(const_cast<float*>(a));
  float4 * glb_b = reinterpret_cast<float4*>(const_cast<float*>(b));

  
  // 256 threads for each block
  const u_int32_t M_tiles = div_ceil(M, M_TILE);
  const u_int32_t N_tiles = div_ceil(N, N_TILE);
  const u_int32_t K_tiles = div_ceil(K, K_TILE);

  // 4 x 2 warps for each block
  const u_int32_t warp_id = threadIdx.x / 32;
  const u_int32_t warp_x = warp_id % 2;
  const u_int32_t warp_y = warp_id / 2;

  // 4 x 8 threads for each warp
  const u_int32_t lane_id = threadIdx.x % 32;
  const u_int32_t lane_x = (lane_id / 2) % 8;
  const u_int32_t lane_y = (lane_id / 16) * 2 + lane_id % 2;

  // block indexs
  const u_int32_t block_x = blockIdx.x;
  const u_int32_t block_y = blockIdx.y;

  const u_int32_t block_dim_x = blockDim.x;
  const u_int32_t block_dim_y = blockDim.y;

  // block strides, a is column-major order, b is row-major order
  const u_int32_t a_block_x_stride = M / FLOAT4_WORD_NUM;
  const u_int32_t a_block_y_stide = 1;
  const u_int32_t b_block_x_stride = 1;
  const u_int32_t b_block_y_stide = N / FLOAT4_WORD_NUM;
  
  // compute store address for shared memory A (column-major)
  const u_int32_t a_smem_st_x = threadIdx.x / 32;
  const u_int32_t a_smem_st_y = threadIdx.x % 32;
  const u_int32_t a_smem_st_x_stride = 32;
  const u_int32_t a_smem_st_y_stride = 1;
  const u_int32_t a_smem_st_addr = a_smem_st_x * a_smem_st_x_stride + a_smem_st_y * a_smem_st_y_stride;

  // compute store address for shared memory B (row-major)
  const u_int32_t b_smem_st_x = threadIdx.x % 32;
  const u_int32_t b_smem_st_y = threadIdx.x / 32;
  const u_int32_t b_smem_st_x_stride = 1;
  const u_int32_t b_smem_st_y_stride = 32;
  const u_int32_t b_smem_st_addr = b_smem_st_x * b_smem_st_x_stride + b_smem_st_y * b_smem_st_y_stride;

  // compute load address for global memory A (column-major)
  const u_int32_t a_glb_ld_x_stride = M / FLOAT4_WORD_NUM;
  const u_int32_t a_glb_ld_y_stride = 1;
  const u_int32_t a_glb_ld_y = block_y * block_dim_y + lane_id;
  // begin x index
  u_int32_t a_glb_ld_x = warp_id;
  // begin global address
  u_int32_t a_glb_ld_addr = a_glb_ld_y * a_glb_ld_y_stride;

  // compute load address for global memory B (column-major)
  const u_int32_t b_glb_ld_x_stride = 1;
  const u_int32_t b_glb_ld_y_stride = N / FLOAT4_WORD_NUM;
  const u_int32_t b_glb_ld_x = block_x * block_dim_x + lane_id;
  // begin y index
  u_int32_t b_glb_ld_y = warp_id;
  // begin global address
  u_int32_t b_glb_ld_addr = b_glb_ld_x* b_glb_ld_x_stride;


  for (u_int32_t k_tile_idx = 0; k_tile_idx < K_tiles; ++k_tile_idx) {
    // [Matrix A]: load from global memory to shared memory
    a_glb_ld_x += k_tile_idx * K_TILE;
    a_glb_ld_addr += a_glb_ld_x * a_glb_ld_x_stride;
    smem_a[a_smem_st_addr] = glb_a[a_glb_ld_addr];

    // [Matrix B]: load from global memory to shared memory
    b_glb_ld_y += k_tile_idx * K_TILE;
    b_glb_ld_addr += b_glb_ld_y * b_glb_ld_y_stride;
    smem_b[b_smem_st_addr] = glb_a[b_glb_ld_addr];

    __syncthreads();
  }
}

void sgemm_128x128x8(const float * a, const float * b, float *c, int M, int N, int K) {
  dim3 block(256);
  int tN = (N - 1) / 128 + 1;
  int tM = (M - 1) / 128 + 1;
  dim3 grid(tN, tM);
  sgemm_128x128x8_kernel<<<grid, block>>>(a, b, c, M, N, K);
}