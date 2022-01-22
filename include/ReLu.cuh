//
// Created by Su on 1/13/2022.
//

#ifndef TAS_RELU_CUH
#define TAS_RELU_CUH

#include "Layer.cuh"
#include "Tensor.cuh"

#include <memory>
#include <string>


class ReLu : public Layer {
public:
    ReLu();
    ~ReLu();

    void forward() override;

    void set_input(const std::shared_ptr<Tensor<float>> &input);
    static std::shared_ptr<Tensor<float>> get_output();

private:
    std::shared_ptr<Tensor<float>> _input, _res;
    int batch_size{};
    void set_input(const std::shared_ptr<float> &input);
};
#endif//TAS_RELU_CUH
