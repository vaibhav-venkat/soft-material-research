#ifndef LINALG_CUDA_H
#define LINALG_CUDA_H

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


_Static_assert(sizeof(cuComplex) == 2 * sizeof(float), "unexpected cuComplex size");
_Static_assert(offsetof(cuComplex, x) == 0 && offsetof(cuComplex, y) == sizeof(float), "unexpected cuComplex fields");
_Static_assert(sizeof(cuDoubleComplex) == 2 * sizeof(double), "unexpected cuDoubleComplex size");
_Static_assert(offsetof(cuDoubleComplex, x) == 0 && offsetof(cuDoubleComplex, y) == sizeof(double), "unexpected cuDoubleComplex fields");

#endif
