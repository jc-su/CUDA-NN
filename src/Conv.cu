//
// Created by Su on 1/11/2022.
//
#include "Conv.cuh"

__global__ void weight2col(const float *w_ptr, float *res_ptr, int N, int C, int H, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int filt_stride = H * W;
    int mat_stride = C * H * W;

    if (i > N) {
        return;
    }
    if (j > mat_stride) {
        return;
    }

    int Ni = (i);
    int Ci = (j) / (H * W);
    int Hi = (j - Ci * H * W) / W;
    int Wi = (j - Ci * H * W - Hi * W);

    res_ptr[i * mat_stride + j] = w_ptr[Ni * mat_stride + Ci * filt_stride + Hi * W + Wi];
}

__global__ void img2col(const float *im_ptr, float *res_ptr, int Nf, int Cf, int Hf, int Wf, int Ho, int Wo, int Hi, int Wi, int batch_size, int stride, int pad, float pad_val = 0) {
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


Conv::Conv(const Tensor<float> &weights, int stride, int pad, bool bias) {
}
Conv::~Conv() = default;
void Conv::set_input(const std::shared_ptr<Tensor<float>> &input) {
}
std::shared_ptr<Tensor<float>> Conv::get_output() {
    return {};
}

void Conv::forward() {}
