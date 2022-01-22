#include "ReLu.cuh"

ReLu::ReLu() = default;

ReLu::~ReLu() = default;

void forward() {}

void ReLu::set_input(const std::shared_ptr<Tensor<float>> &input) {
}
std::shared_ptr<Tensor<float>> ReLu::get_output() {
    return nullptr;
}
void ReLu::forward() {
}
void ReLu::set_input(const std::shared_ptr<float> &input) {
}
