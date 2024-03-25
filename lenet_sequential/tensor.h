#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef> // For float

class Tensor {
public:
    Tensor();
    Tensor(size_t h, size_t w, size_t d);
    ~Tensor(); 

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    float& operator()(size_t x, size_t y, size_t z);
    const float& operator()(size_t x, size_t y, size_t z) const;

    size_t getHeight() const;
    size_t getWidth() const;
    size_t getDepth() const;

    void fill(float value);
    void print() const;

private:
    size_t height, width, depth;
    std::vector<float> data;
};

#endif // TENSOR_H