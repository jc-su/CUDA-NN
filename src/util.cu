template<typename T>
__global__ void add_ker(T *src1, T *src2, T *dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    dst[i] = src1[i] + src2[i];
}

template<typename T>
void cuda_add(T *src1, T *src2, T *res, int N) {
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N / cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    add_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}

template<typename T>
__global__ void sub_ker(T *src1, T *src2, T *dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    dst[i] = src1[i] - src2[i];
}

template<typename T>
void cuda_sub(T *src1, T *src2, T *res, int N) {
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N / cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    sub_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}


template<typename T>
__global__ void mul_ker(T *src1, T *src2, T *dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    dst[i] = src1[i] * src2[i];
}

template<typename T>
void cuda_mul(T *src1, T *src2, T *res, int N) {
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N / cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    mul_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}

template<typename T>
__global__ void div_ker(T *src1, T *src2, T *dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    dst[i] = src1[i] / src2[i];
}

template<typename T>
void cuda_div(T *src1, T *src2, T *res, int N) {
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;

    num_blocks_x = N / cell_size + (N % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);
    div_ker<T><<<grid_size, block_size>>>(src1, src2, res, N);
}


template<typename T>
__global__ void transpose_ker(T *src_ptr, T *dst_ptr, int *src_dims, int *strides, int *reorder, int *new_strides, int Ndims, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    int new_idx[10];
    int acc = 0;
    for (int k = 0; k < Ndims; ++k) {
        int cur_i = (i - acc) / strides[k];
        acc += cur_i * strides[k];

        new_idx[reorder[k]] = cur_i;
    }

    int new_i = 0;
    for (int k = 0; k < Ndims; ++k) {
        new_i += new_strides[k] * new_idx[k];
    }

    dst_ptr[new_i] = src_ptr[i];
}

template<typename T>
void cuda_transpose(T *src_ptr, T *dst_ptr, int *src_dims, int *strides, int *reorder, int *new_strides, int Ndims, int N) {
    int cell_size = 32;
    dim3 block_size;
    dim3 grid_size;
    int num_blocks_x;
    num_blocks_x = (N) / cell_size + ((N) % cell_size != 0);
    block_size = dim3(cell_size);
    grid_size = dim3(num_blocks_x);

    transpose_ker<<<grid_size, block_size>>>(src_ptr, dst_ptr, src_dims, strides, reorder, new_strides, Ndims, N);
}

__global__ void im2col(float *im_ptr, float *res_ptr, int Nf, int Cf, int Hf, int Wf, int Ho, int Wo, int Hi, int Wi, int batch_size, int stride, int pad, float pad_val = 0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_mat_stride = Cf * Hi * Wi;
    int o_mat_stride = Cf * Hf * Wf;
    int total_scals = Ho * Wo * batch_size;

    if (i >= total_scals) {
        return;
    }
    if (j >= Hf * Wf * Cf) {
        return;
    }

    int Ri = ((i % (Ho * Wo)) / Wo);
    int Rj = ((i % (Ho * Wo)) % Wo);
    int ci = j / (Hf * Wf);
    int K_ind_i = (j - ci * Hf * Wf) / Wf;
    int K_ind_j = (j - ci * Hf * Wf) % Wf;

    int hi = stride * Ri + K_ind_i;
    int wi = stride * Rj + K_ind_j;
    int ni = i / (Ho * Wo);// batch

    bool is_pad = (hi < pad) || (wi < pad) || (hi >= Hi + pad) || (wi >= Wi + pad);

    if (!is_pad) {
        hi -= pad;
        wi -= pad;
        res_ptr[i * o_mat_stride + j] = im_ptr[ni * i_mat_stride + ci * Hi * Wi + hi * Wi + wi];
    } else {
        res_ptr[i * o_mat_stride + j] = pad_val;
    }
}

template void cuda_add<float>(float *, float *, float *, int);
template void cuda_sub<float>(float *, float *, float *, int);
template void cuda_mul<float>(float *, float *, float *, int);
template void cuda_div<float>(float *, float *, float *, int);

template void cuda_add<int>(int *, int *, int *, int);
template void cuda_sub<int>(int *, int *, int *, int);
template void cuda_mul<int>(int *, int *, int *, int);
template void cuda_div<int>(int *, int *, int *, int);

template void cuda_transpose<float>(float *, float *, int *, int *, int *, int *, int, int);
template void cuda_transpose<int>(int *, int *, int *, int *, int *, int *, int, int);