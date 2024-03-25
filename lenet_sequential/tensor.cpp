#include "tensor.h"
#include <iostream>
#include <cassert>
#include <algorithm> // For std::fill

Tensor::Tensor() : height(0), width(0), depth(0) {}

Tensor::Tensor(size_t h, size_t w, size_t d) : height(h), width(w), depth(d), data(h * w * d, 0) {}

Tensor::~Tensor() {}

Tensor::Tensor(const Tensor& other) : height(other.height), width(other.width), depth(other.depth), data(other.data) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        height = other.height;
        width = other.width;
        depth = other.depth;
        data = other.data;
    }
    return *this;
}

float& Tensor::operator()(size_t x, size_t y, size_t z) {
    assert(x < height && y < width && z < depth);
    return data[x * width * depth + y * depth + z];
}

const float& Tensor::operator()(size_t x, size_t y, size_t z) const {
    assert(x < height && y < width && z < depth);
    return data[x * width * depth + y * depth + z];
}

size_t Tensor::getHeight() const { return height; }
size_t Tensor::getWidth() const { return width; }
size_t Tensor::getDepth() const { return depth; }

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::print() const {
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            for (size_t k = 0; k < depth; ++k) {
                std::cout << (*this)(i, j, k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
