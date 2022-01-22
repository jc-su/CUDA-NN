#ifndef TAS_FC_CUH
#define TAS_FC_CUH

#include "Layer.cuh"
#include "Tensor.cuh"

#include <memory>
#include <string>


class FC : public Layer {
public:
    explicit FC(const Tensor<float>& weights, bool _bias = true);

    ~FC();

    void forward() override;

    void set_input(const std::shared_ptr<Tensor<float>>& input);

    static std::shared_ptr<Tensor<float>> get_output();

    static int get_output_dim();

private:
    int batch_size, input_dim, output_dim;
    std::shared_ptr<Tensor<float>> _input, _w, _b, _res, _tmp;
    Tensor<float> weights;
    std::vector<float> data_b;
    bool _bias;
};

#endif//TAS_FC_CUH
