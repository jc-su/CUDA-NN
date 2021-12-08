#ifndef TAS_POOL_CUH
#define TAS_POOL_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <memory>
#include <string>


class Pool : public Layer {
public:
    Pool(int ker_size = 2, int stride = 1, int pad = 0);

    ~Pool();

    void forward();

    void set_input(std::shared_ptr<Tensor<float>> input);

    std::shared_ptr<Tensor<float>> get_output();

private:
    std::shared_ptr<Tensor<float>> _input, _res;
    int batch_size, _stride, _pad, H, W, C;
    int Hi, Wi, Ho, Wo;
};

#endif//TAS_POOL_CUH
