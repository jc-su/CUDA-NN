#include <utility>

#include "Tensor.cuh"

template<typename T>
Tensor<T>::Tensor(Size size_p) : _size(std::move(size_p)) {
    _count = 1;
    for (int i : _size) {
        _count *= i;
    }
    cudaMalloc(&_ptr, _count * sizeof(T));
}

template<typename T>
Tensor<T>::~Tensor() {
    cudaFree(_ptr);
}

template<typename T>
int Tensor<T>::n_dim() const {
    return _n_dim;
}

template<typename T>
int Tensor<T>::count() const {
    return _count;
}

template<typename T>
void Tensor<T>::from_cpu(T *ptr) {
    cudaMemcpy(_ptr, ptr, _count * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void Tensor<T>::to_cpu(T *ptr) {
    cudaMemcpy(ptr, _ptr, _count * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
const Size &Tensor<T>::size() {
    return _size;
}

template<typename T>
Tensor<T> &Tensor<T>::reshape(const Size &new_size) {
    int new_count = 1;
    for (int i : new_size) {
        new_count *= i;
    }
    if (new_count != _count) {
        throw std::runtime_error("reshape wrong size");
    }
    _size = new_size;
    _count = new_count;
    _n_dim = _size.size();
    return *this;
}

template<typename T>
Tensor<T> *Tensor<T>::transpose(Tensor<T> *src, Tensor<T> *dst, const std::vector<int> &order) {
    // create arrays for reshape
    std::shared_ptr<Tensor<int>> _dims, _reorder, _strides, _new_strides;
    int n_dims = src->n_dim();

    Size dims_cpu(src->size());
    _dims = std::shared_ptr<Tensor<int>>(new Tensor<int>({n_dims}));
    _dims->from_cpu(dims_cpu.data());


    Size strides_cpu(n_dims);
    int cnt = 1;
    for (int i = n_dims - 1; i >= 0; --i) {
        strides_cpu[i] = cnt;
        cnt *= dims_cpu[i];
    }
    _strides = std::shared_ptr<Tensor<int>>(new Tensor<int>({n_dims}));
    _strides->from_cpu(strides_cpu.data());

    Size reorder_cpu(order);
    _reorder = std::shared_ptr<Tensor<int>>(new Tensor<int>({n_dims}));
    _reorder->from_cpu(reorder_cpu.data());

    Size new_strides_cpu(n_dims);
    cnt = 1;
    for (int i = n_dims - 1; i >= 0; --i) {
        new_strides_cpu[i] = cnt;
        cnt *= dims_cpu[reorder_cpu[i]];
    }
    _new_strides = std::shared_ptr<Tensor<int>>(new Tensor<int>({n_dims}));
    _new_strides->from_cpu(new_strides_cpu.data());

    cuda_transpose(src->_ptr, dst->_ptr, _dims->_ptr, _strides->_ptr, _reorder->_ptr, _new_strides->_ptr, n_dims,
                   src->count());

    return dst;
}

template<typename T>
Tensor<T> &Tensor<T>::operator+=(const Tensor<T> &src2) {
    if (this->count() != src2.count()) {
        throw std::runtime_error("different size in Tensor::add_inplace");
    }
    cuda_add(this->_ptr, src2._ptr, this->_ptr, this->count());
    return *this;
}

template<typename T>
Tensor<T> &Tensor<T>::operator-=(const Tensor<T> &src2) {
    if (this->count() != src2.count()) {
        throw std::runtime_error("different size in Tensor::sub_inplace");
    }
    cuda_sub(this->_ptr, src2._ptr, this->_ptr, this->count());
    return *this;
}

template<typename T>
Tensor<T> &Tensor<T>::operator*=(const Tensor<T> &src2) {
    if (this->count() != src2.count()) {
        throw std::runtime_error("different size in Tensor::mul_inplace");
    }
    cuda_mul(this->_ptr, src2._ptr, this->_ptr, this->count());
    return *this;
}

template<typename T>
Tensor<T> &Tensor<T>::operator/=(const Tensor<T> &src2) {
    if (this->count() != src2.count()) {
        throw std::runtime_error("different size in Tensor::div_inplace");
    }
    cuda_div(this->_ptr, src2._ptr, this->_ptr, this->count());
    return *this;
}

template class Tensor<float>;

// for int8
template class Tensor<int>;