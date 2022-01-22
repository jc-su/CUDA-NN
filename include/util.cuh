#ifndef TAS_UTIL_CUH
#define TAS_UTIL_CUH

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
transpose_ker(T *src_ptr, T *dst_ptr, int *src_dims, int *strides, int *reorder, int *new_strides, int n_dims);

template<typename T>
void cuda_transpose(T *src_ptr, T *dst_ptr, int *src_dims, int *strides, int *reorder, int *new_strides, int Ndims, int N);

__global__ void im2col(float *im_ptr, float *res_ptr, int Nf, int Cf, int Hf, int Wf, int Ho, int Wo, int Hi, int Wi, int batch_size, int stride, int pad, float pad_val = 0);

__global__ void w2col(float* w_ptr, float* res_ptr, int N, int C, int H, int W);

#endif//TAS_UTIL_CUH
