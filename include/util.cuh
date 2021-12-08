#ifndef TAS_UTIL_CUH
#define TAS_UTIL_CUH

#include <cublas.h>
#include <cublas_v2.h>

__global__ void debug_ker(float *ptr, int addr);

void debug_array(float *arr, int N);

template<typename T>
void cuda_add(T *src1, T *src2, T *res, int N);

template<typename T>
void cuda_sub(T *src1, T *src2, T *res, int N);

template<typename T>
void cuda_mul(T *src1, T *src2, T *res, int N);

template<typename T>
void cuda_div(T *src1, T *src2, T *res, int N);

template<typename T>
__global__ void
transpose_ker(T *src_ptr, T *dst_ptr, int *src_dims, int *strides, int *reorder, int *new_strides, int Ndims);

template<typename T>
void cuda_transpose(T *src_ptr, T *dst_ptr, int *src_dims, int *strides, int *reorder, int *new_strides, int Ndims, int N);

#endif//TAS_UTIL_CUH
