#include <cstdint>
#include <iostream>

__global__ void smem_1(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid < 16) {
    reinterpret_cast<uint2 *>(a)[tid] =
        reinterpret_cast<const uint2 *>(smem)[tid];
  }
}

__global__ void smem_2(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  if (tid < 15 || tid == 16) {
    reinterpret_cast<uint2 *>(a)[tid] =
        reinterpret_cast<const uint2 *>(smem)[tid == 16 ? 15 : tid];
  }
}

__global__ void smem_3(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid / 2];
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
    addr = tid / 2;
  } else {
    addr = (tid / 4) * 2 + (tid % 4) % 2;
  }
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[addr];
}

__global__ void smem_5(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid % 16];
}

__global__ void smem_6(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = (tid / 4) * 2 + (tid % 4) % 2;
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[addr];
}

__global__ void smem_7(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = (tid / 16) * 2 + (tid % 2);
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[addr];
}

__global__ void smem_8(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  uint32_t addr = (tid / 16) * 2 + (tid % 2);
  reinterpret_cast<uint4 *>(a)[tid] =
      reinterpret_cast<const uint4 *>(smem)[addr];
}

int main() {
  uint32_t *d_a;
  cudaMalloc(&d_a, sizeof(uint32_t) * 128);
  // smem_1<<<1, 32>>>(d_a);
  // smem_2<<<1, 32>>>(d_a);
  // smem_3<<<1, 32>>>(d_a);
  // smem_4<<<1, 32>>>(d_a);
  // smem_5<<<1, 32>>>(d_a);
  // smem_6<<<1, 32>>>(d_a);
  smem_7<<<1, 32>>>(d_a);
  smem_8<<<1, 32>>>(d_a);

  // for (int tid = 0; tid < 32; ++tid) {
  //   uint32_t addr;
  //   // if (tid < 16) {
  //   //   addr = tid / 2;
  //   // } else {
  //   //   addr = (tid / 4) * 2 + (tid % 4) % 2;
  //   // }
  //   addr = (tid / 16) * 2 + (tid % 2);
  //   std::cout << "tid: " << tid << " , addr: " << addr << std::endl; 
  // }

  cudaFree(d_a);
  cudaDeviceSynchronize();
  return 0;
}