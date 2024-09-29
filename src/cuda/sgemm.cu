
#include <sgemm.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
__device__ const u_int32_t M_TILE = 128;
#define N_TILE 128
#define K_TILE 8

#define FLOAT4_WORD_NUM (sizeof(float4) / sizeof(float))

#define LANE_HEIGHT 4 // 4 sizeof(float4)
#define LANE_WIDTH 8 // 8 sizeof(float4)

#define WARP_HEIGHT 2 * LANE_HEIGHT
#define WARP_WIDTH 2 * LANE_WIDTH

#define WARP_X_GLB_STRIDE 64 / FLOAT4_WORD_NUM
#define WARP_Y_GLB_STRIDE 2 * WARP_X_GLB_STRIDE

__device__ constexpr u_int32_t M_TILE_FLOAT4  = M_TILE / FLOAT4_WORD_NUM;
__device__ constexpr u_int32_t N_TILE_FLOAT4  = N_TILE / FLOAT4_WORD_NUM;

__device__ constexpr u_int32_t A_SMEM_OFFSET_STRIDE = M_TILE_FLOAT4 * K_TILE;
__device__ constexpr u_int32_t B_SMEM_OFFSET_STRIDE = N_TILE_FLOAT4 * K_TILE;

inline __device__ __host__ u_int32_t div_ceil(u_int32_t a, u_int32_t b) {
    return (a - 1) / b + 1;
}

inline __device__ void mma4x4(const float4 & fragmentA, const float4 & fragmentB, float4 * tc){
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

__global__ void sgemm_128x128x8_kernel(const float * a, const float * b, float *c, int M, int N, int K) {
  // shared memory
  __shared__ __align__(32 *sizeof(float)) float smem[2 * M_TILE * K_TILE + 2 * K_TILE * N_TILE];

  float4 * smem_a = reinterpret_cast<float4*>(smem);
  float4 * smem_b = reinterpret_cast<float4*>(smem + 2 * M_TILE * K_TILE);

  float4 * glb_a = reinterpret_cast<float4*>(const_cast<float*>(a));
  float4 * glb_b = reinterpret_cast<float4*>(const_cast<float*>(b));
  float4 * glb_c = reinterpret_cast<float4*>(const_cast<float*>(c));

  
  // 256 threads for each block
  const u_int32_t M_tiles = div_ceil(M, M_TILE);
  const u_int32_t N_tiles = div_ceil(N, N_TILE);
  const u_int32_t K_tiles = div_ceil(K, K_TILE);
  // printf("K_tiles: %d\n", K_tiles);

  // 4 x 2 warps for each block
  const u_int32_t warp_id = threadIdx.x / 32;
  const u_int32_t warp_x = warp_id % 2;
  const u_int32_t warp_y = warp_id / 2;

  // 4 x 8 threads for each warp
  const u_int32_t lane_id = threadIdx.x % 32;
  const u_int32_t lane_x = (lane_id / 2) % 8;
  const u_int32_t lane_y = (lane_id / 16) * 2 + lane_id % 2;


  const u_int32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;

  const u_int32_t block_x = blockIdx.x;
  const u_int32_t block_y = blockIdx.y;

  constexpr u_int32_t block_dim_x = N_TILE_FLOAT4;
  constexpr u_int32_t block_dim_y = M_TILE_FLOAT4;

  // block strides, a is column-major order, b is row-major order
  const u_int32_t a_block_x_stride = M / FLOAT4_WORD_NUM;
  const u_int32_t a_block_y_stide = 1;
  const u_int32_t b_block_x_stride = 1;
  const u_int32_t b_block_y_stide = N / FLOAT4_WORD_NUM;

  const u_int32_t c_glb_st_x_stride = 1;
  const u_int32_t c_glb_st_y_stride = N / FLOAT4_WORD_NUM;
  
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
  u_int32_t a_glb_ld_addr = a_glb_ld_y * a_glb_ld_y_stride + warp_id * a_glb_ld_x_stride;

  // compute load address for global memory B (column-major)
  const u_int32_t b_glb_ld_x_stride = 1;
  const u_int32_t b_glb_ld_y_stride = N / FLOAT4_WORD_NUM;
  const u_int32_t b_glb_ld_x = block_x * block_dim_x + lane_id;
  // begin y index
  u_int32_t b_glb_ld_y = warp_id;
  // begin global address
  u_int32_t b_glb_ld_addr = b_glb_ld_x * b_glb_ld_x_stride + b_glb_ld_y * b_glb_ld_y_stride;

  // used for double buffer
  u_int32_t A_SM_OFFSET = 0;
  u_int32_t B_SM_OFFSET = 0;

  // [Matrix A]: load from global memory to shared memory
  smem_a[a_smem_st_addr] = glb_a[a_glb_ld_addr];

  // [Matrix B]: load from global memory to shared memory
  smem_b[b_smem_st_addr] = glb_b[b_glb_ld_addr];

  __syncthreads();

  A_SM_OFFSET ^= 1;
  B_SM_OFFSET ^= 1;

  float4 frag_a[2];
  float4 frag_b[2];
  float4 tc_zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 tc4x4[4][4] = {{make_float4(0.0f, 0.0f, 0.0f, 0.0f)}};
  // printf("block_x: %d, block_y: %d\n", block_x, block_y);

  for (u_int32_t k_tile_idx = 1; k_tile_idx < K_tiles; ++k_tile_idx) {
    // [Matrix A]: load from global memory to shared memory
    a_glb_ld_addr += K_TILE * a_glb_ld_x_stride;
    smem_a[A_SM_OFFSET * M_TILE_FLOAT4 * K_TILE + a_smem_st_addr] = glb_a[a_glb_ld_addr];

    // [Matrix B]: load from global memory to shared memory
    b_glb_ld_addr += K_TILE * b_glb_ld_y_stride;
    smem_b[B_SM_OFFSET * N_TILE_FLOAT4 * K_TILE + b_smem_st_addr] = glb_b[b_glb_ld_addr];

    A_SM_OFFSET ^= 1;
    B_SM_OFFSET ^= 1;

    // compute fragment

    #pragma unroll
    for (int k_tile_smem = 0; k_tile_smem < K_TILE; ++k_tile_smem) {
      u_int32_t a_smem_ld_0 = A_SM_OFFSET * A_SMEM_OFFSET_STRIDE + k_tile_smem * M_TILE_FLOAT4 + warp_y * WARP_HEIGHT + lane_y;
      u_int32_t a_smem_ld_1 = A_SM_OFFSET * A_SMEM_OFFSET_STRIDE + k_tile_smem * M_TILE_FLOAT4 + warp_y * WARP_HEIGHT + lane_y + LANE_HEIGHT;
      u_int32_t b_smem_ld_0 = B_SM_OFFSET * B_SMEM_OFFSET_STRIDE + k_tile_smem * N_TILE_FLOAT4 + warp_x * WARP_WIDTH + lane_x;
      u_int32_t b_smem_ld_1 = B_SM_OFFSET * B_SMEM_OFFSET_STRIDE + k_tile_smem * N_TILE_FLOAT4 + warp_x * WARP_WIDTH + lane_x + LANE_WIDTH;

      frag_a[0] = smem_a[a_smem_ld_0];
      frag_a[1] = smem_a[a_smem_ld_1];
      frag_b[0] = smem_b[b_smem_ld_0];
      frag_b[1] = smem_b[b_smem_ld_1];

      mma4x4(frag_a[0], frag_b[0], tc4x4[0]);
      mma4x4(frag_a[0], frag_b[1], tc4x4[1]);
      mma4x4(frag_a[1], frag_b[0], tc4x4[2]);
      mma4x4(frag_a[1], frag_b[1], tc4x4[3]);

    }

    __syncthreads();
  
  }

  A_SM_OFFSET ^= 1;
  B_SM_OFFSET ^= 1;

  #pragma unroll
  for (int k_tile_smem = 0; k_tile_smem < K_TILE; ++k_tile_smem) {
    u_int32_t a_smem_ld_0 = A_SM_OFFSET * A_SMEM_OFFSET_STRIDE + k_tile_smem * M_TILE_FLOAT4 + warp_y * WARP_HEIGHT + lane_y;
    u_int32_t a_smem_ld_1 = A_SM_OFFSET * A_SMEM_OFFSET_STRIDE + k_tile_smem * M_TILE_FLOAT4 + warp_y * WARP_HEIGHT + lane_y + LANE_HEIGHT;
    u_int32_t b_smem_ld_0 = B_SM_OFFSET * B_SMEM_OFFSET_STRIDE + k_tile_smem * N_TILE_FLOAT4 + warp_x * WARP_WIDTH + lane_x;
    u_int32_t b_smem_ld_1 = B_SM_OFFSET * B_SMEM_OFFSET_STRIDE + k_tile_smem * N_TILE_FLOAT4 + warp_x * WARP_WIDTH + lane_x + LANE_WIDTH;

    frag_a[0] = smem_a[a_smem_ld_0];
    frag_a[1] = smem_a[a_smem_ld_1];
    frag_b[0] = smem_b[b_smem_ld_0];
    frag_b[1] = smem_b[b_smem_ld_1];

    mma4x4(frag_a[0], frag_b[0], tc4x4[0]);
    mma4x4(frag_a[0], frag_b[1], tc4x4[1]);
    mma4x4(frag_a[1], frag_b[0], tc4x4[2]);
    mma4x4(frag_a[1], frag_b[1], tc4x4[3]);

  }

  u_int32_t c_glb_st_x_0 = block_x * block_dim_x + warp_x * WARP_WIDTH + lane_x;
  u_int32_t c_glb_st_x_1 = block_x * block_dim_x + warp_x * WARP_WIDTH + lane_x + LANE_WIDTH;
  u_int32_t c_glb_st_y_0 = block_y * block_dim_y + warp_y * WARP_HEIGHT + lane_y;
  u_int32_t c_glb_st_y_1 = block_y * block_dim_y + warp_y * WARP_HEIGHT + lane_y + LANE_HEIGHT;

  u_int32_t c_glb_st_0 = c_glb_st_y_0 * c_glb_st_y_stride * FLOAT4_WORD_NUM + c_glb_st_x_0 * c_glb_st_x_stride;
  u_int32_t c_glb_st_1 = c_glb_st_y_0 * c_glb_st_y_stride * FLOAT4_WORD_NUM + c_glb_st_x_1 * c_glb_st_x_stride;
  u_int32_t c_glb_st_2 = c_glb_st_y_1 * c_glb_st_y_stride * FLOAT4_WORD_NUM + c_glb_st_x_0 * c_glb_st_x_stride;
  u_int32_t c_glb_st_3 = c_glb_st_y_1 * c_glb_st_y_stride * FLOAT4_WORD_NUM + c_glb_st_x_1 * c_glb_st_x_stride;

  for (u_int32_t row = 0; row < FLOAT4_WORD_NUM; ++ row) {
    glb_c[c_glb_st_0 + row * c_glb_st_y_stride] = tc4x4[0][row];
    glb_c[c_glb_st_1 + row * c_glb_st_y_stride] = tc4x4[1][row];
    glb_c[c_glb_st_2 + row * c_glb_st_y_stride] = tc4x4[2][row];
    glb_c[c_glb_st_3 + row * c_glb_st_y_stride] = tc4x4[3][row];
  }

}

void sgemm_128x128x8(const float * a, const float * b, float *c, int M, int N, int K) {
  dim3 block(256);
  int tN = (N - 1) / 128 + 1;
  int tM = (M - 1) / 128 + 1;
  dim3 grid(tN, tM);
  sgemm_128x128x8_kernel<<<grid, block>>>(a, b, c, M, N, K);
}