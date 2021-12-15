#ifndef TAS_LINEAR_CUH
#define TAS_LINEAR_CUH

#include "Layer.hpp"
#include "Tensor.hpp"

#include <memory>
#include <string>


class Linear : public Layer {
public:
    Linear(Tensor<float> weights, bool bias = true);

    ~Linear();

    void forward();

    void set_input(std::shared_ptr<Tensor<float>> input);

    std::shared_ptr<Tensor<float>> get_output();

    int get_output_dim();

private:
    int batch_size, input_dim, output_dim;
    std::shared_ptr<Tensor<float>> _input, _w, _b, _res, _tmp;
    Tensor<float> weights;
    std::vector<float> data_b;
    bool _bias;
};

#endif//TAS_LINEAR_CUH
