//
// Created by eugene on 2021/4/30.
//

#ifndef __DEF_H__
#define __DEF_H__

#ifdef __NVCC__
#include <cuda.h>
#include <cuda_runtime.h>

#define __qualifier__ __host__ __device__
#else
#define __qualifier__
#endif

#endif
