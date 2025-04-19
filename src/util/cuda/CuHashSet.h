#pragma once

namespace cuda_hashset {
    struct HashSet {
        int *occupied;
        int size;

        HashSet(int size);
        ~HashSet();
    };

    void initialize(int* occupied, int size);
    __global__ void initializeHashSet(int* occupied, int size);
    void allocateHashSet(int* occupied, int size);
    __device__ unsigned int hash(int key, int size);
    __device__ bool insert(int* occupied, int size, int key);
}
