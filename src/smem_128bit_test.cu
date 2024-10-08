#include <cstdint>
#include <iostream>

__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 15 || tid == 16) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}

// dont know why only 1 wavefront
__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid == 0 || tid == 15) {
    reinterpret_cast<uint4 *>(a)[tid] =
        reinterpret_cast<const uint4 *>(smem)[4];
  }
}

__global__ void smem_3(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint4 *>(a)[tid] = reinterpret_cast<const uint4 *>(
      smem)[(tid / 8) * 2 + ((tid % 8) / 2) % 2];
}

__global__ void smem_4(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr;
  if (tid < 16) {
    addr = (tid / 8) * 2 + ((tid % 8) / 2) % 2;
  } else {
    addr = (tid / 8) * 2 + ((tid % 8) % 2);
  }
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

__global__ void smem_5(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[(tid / 16) * 4 + (tid % 16) / 8 + (tid % 8) / 4 * 8];
}

__global__ void smem_6(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = (tid / 16) * 4 + (tid % 16 / 8) * 8;
  if (tid < 16) {
    addr += (tid % 4 / 2) * 2;
  } else {
    addr += (tid % 4 % 2) * 2;
  }
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

__global__ void smem_7(uint32_t *a) {
  __shared__ uint32_t smem[32];
  uint32_t tid = threadIdx.x;
  smem[tid] = tid;
  __syncthreads();
  uint32_t addr = (tid / 2) % 8;

  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

__global__ void smem_8(uint32_t *a) {
  __shared__ uint32_t smem[32];
  uint32_t tid = threadIdx.x;
  smem[tid] = tid;
  __syncthreads();
  uint32_t addr = (tid / 16) * 2 + tid % 2;

  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

int main() {
  uint32_t *d_a;
  cudaMalloc(&d_a, sizeof(uint32_t) * 128);
  smem_1<<<1, 32>>>(d_a);
  smem_2<<<1, 32>>>(d_a);
  smem_3<<<1, 32>>>(d_a);
  smem_4<<<1, 32>>>(d_a);
  smem_5<<<1, 32>>>(d_a);
  smem_6<<<1, 32>>>(d_a);
  smem_7<<<1, 32>>>(d_a);
  smem_8<<<1, 32>>>(d_a);
  cudaFree(d_a);
  cudaDeviceSynchronize();

  for (int id = 0; id < 128; ++id) {
    int height = 16;
    int flag = id / (height * 2) % 2;
    const u_int32_t lane_x = ((flag + 1) % 2) * (id / 2) % height + (flag % 2) * (height - 1 - (id / 2) % height);
    const u_int32_t lane_y = (id / (height * 2)) * 2 + id % 2;
    std::cout << "id: " << id << " ( " << lane_y << ", " << lane_x << ")" << "  flag: " << flag <<std::endl;
  }


  return 0;
}