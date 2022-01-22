#ifndef TAS_TENSOR_CUH
#define TAS_TENSOR_CUH

#include <iostream>
#include <memory>
#include <stdexcept>
#include <util.cuh>
#include <vector>

typedef std::vector<int>
        Size;

template<typename T>
class Tensor {

public:
    explicit Tensor(Size size_p);

    virtual ~Tensor();

    int count() const;

    int n_dim() const;

    const Size &size();

    void from_cpu(T *ptr);

    void to_cpu(T *ptr);

    Tensor &reshape(const Size &new_size);

    static Tensor *transpose(Tensor *src, Tensor *dst, const std::vector<int> &order);

    Tensor &operator+=(const Tensor &src2);

    Tensor &operator-=(const Tensor &src2);

    Tensor &operator*=(const Tensor &src2);

    Tensor &operator/=(const Tensor &src2);

    T *_ptr;

private:
    Size _size;
    int _count;
    int _n_dim;
};

#endif//TAS_TENSOR_CUH
