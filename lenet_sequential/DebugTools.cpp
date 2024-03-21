#include "DebugTools.h"
#include <iostream>

// Function implementation
void displayFirstImageAndLabel(const std::vector<float>& images, const std::vector<unsigned char>& labels, int imageSize) {
    if (!images.empty() && !labels.empty()) {
        std::cout << "First label: " << static_cast<int>(labels[0]) << std::endl;
        std::cout << "First image:\n";
        for (int i = 0; i < imageSize; ++i) {
            std::cout << static_cast<int>(images[i]) << " ";
            if ((i + 1) % 28 == 0) std::cout << std::endl; // New line after each 28 pixels
        }
    } else {
        std::cerr << "Failed to load the data properly." << std::endl;
    }
}
