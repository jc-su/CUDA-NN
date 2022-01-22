#ifndef TAS_CONV_CUH
#define TAS_CONV_CUH

#include "Layer.cuh"
#include "Tensor.cuh"

#include <memory>
#include <string>


class Conv : public Layer {
public:
    explicit Conv(const Tensor<float> &weights, int stride = 1, int pad = 0, bool bias = true);

    ~Conv();

    void forward() override;

    void set_input(const std::shared_ptr<Tensor<float>> &input);

    static std::shared_ptr<Tensor<float>> get_output();

private:
    std::shared_ptr<Tensor<float>> _input, _w, _b, _res;
    std::shared_ptr<Tensor<float>> _input_col, _weight_col, _bias_col, _tmp;
    std::vector<float> data_b;
    int Hi, Wi, Ho, Wo;
    int batch_size, N, C, H, W, _pad, _stride;
    int m, n, k;
    bool input_set, _bias;
};

#endif//TAS_CONV_CUH
