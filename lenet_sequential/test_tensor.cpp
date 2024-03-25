#include "tensor.h"
#include <iostream>

int main() {
    // Create a 3x4x2 Tensor
    Tensor t(3, 1, 2);

    // Fill the tensor with some values
    for (size_t i = 0; i < t.getHeight(); ++i) {
        for (size_t j = 0; j < t.getWidth(); ++j) {
            for (size_t k = 0; k < t.getDepth(); ++k) {
                t(i, j, k) = static_cast<float>(i * 100 + j * 10 + k);
            }
        }
    }   

    // Print the tensor to check the values
    std::cout << "Initial Tensor:" << std::endl;
    t.print();

    

    // Print the tensor again to see the changes
    std::cout << "\nModified Tensor:" << std::endl;
    // Change a specific value
    t(1, 2, 1) = 888.88f;
    t.print();

    // Fill the tensor with a specific value
    t.fill(5.55f);

    // Print the tensor again to see the changes
    std::cout << "\nModified Tensor:" << std::endl;
    t.print();
    
    // print hello world
    std::cout << "Hello, World!" << std::endl;
    return 0;
}