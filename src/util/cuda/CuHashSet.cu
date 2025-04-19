#include "src/pch.h"

#include "src/util/cuda/CuHashSet.h"

__global__ void cuda_hashset::initializeHashSet(int* occupied, int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < size){
        occupied[idx] = -1;
    }
}

cuda_hashset::HashSet::HashSet(int size) {
    this->size = size;
    cudaMalloc(&occupied, size * sizeof(int));
    initialize(occupied, size);
    cudaDeviceSynchronize();
}

void cuda_hashset::initialize(int* occupied, int size){
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    initializeHashSet<<<blocksPerGrid, threadsPerBlock>>>(occupied, size);
}


cuda_hashset::HashSet::~HashSet() {
    cudaFree(occupied);
}

__device__ unsigned int cuda_hashset::hash(int key, int size) {
    return key % size;
}

__device__ bool cuda_hashset::insert(int* occupied, int size, int key) {
    int idx = hash(key, size);
    while (true) {
        int expected = -1;
        if (atomicCAS(&occupied[idx], expected, key) == expected) {
            return true;
        } else {
            if (occupied[idx] == key) {
                return false;
            }
        }
        idx = (idx + 1) % size;
    }
}
