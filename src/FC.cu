//
// Created by Su on 1/11/2022.
//
#include "FC.cuh"

FC::FC(const Tensor<float>& weights, bool bias) {
}
FC::~FC() = default;
void FC::set_input(const std::shared_ptr<Tensor<float>>& input) {
}
int FC::get_output_dim() {
    return 0;
}
void FC::forward() {
}
std::shared_ptr<Tensor<float>> FC::get_output() {
    return {};
}
