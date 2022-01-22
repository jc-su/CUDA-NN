#ifndef TAS_POOL_CUH
#define TAS_POOL_CUH

#include "Layer.cuh"
#include "Tensor.cuh"

#include <memory>
#include <string>


class Pooling : public Layer {
public:
    explicit Pooling(int ker_size = 2, int stride = 1, int pad = 0);

    ~Pooling();

    void forward() override;

    void set_input(const std::shared_ptr<Tensor<float>>& input);

    std::shared_ptr<Tensor<float>> get_output();

private:
    std::shared_ptr<Tensor<float>> _input, _res;
    int batch_size{}, _stride, _pad, H, W, C{};
    int Hi{}, Wi{}, Ho{}, Wo{};
};

#endif//TAS_POOL_CUH
