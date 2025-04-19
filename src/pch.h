// pch.h
#ifndef PCH_H
#define PCH_H

#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "src/pcrenderer/pdflow/PDFlow.h"

// Libtorch Headers
#undef slots
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/optim.h>
#include <torch/optim/adam.h>
#include <torch/optim/sgd.h>
#include <c10/cuda/CUDAGuard.h>
#define slots Q_SLOTS

#endif // PCH_H
