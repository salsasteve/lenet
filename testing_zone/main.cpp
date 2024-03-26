#include <iostream>

using namespace std;

int main() {
  // Create a 3D tensor with dimensions 2x3x4
  Eigen::Tensor<int, 3> my_tensor(2, 3, 4);

  // Initialize the tensor with values
  for (int i = 0; i < my_tensor.dimension(0); i++) {
    for (int j = 0; j < my_tensor.dimension(1); j++) {
      for (int k = 0; k < my_tensor.dimension(2); k++) {
        my_tensor(i, j, k) = i + j + k;
      }
    }
  }

  // Print the tensor
  cout << my_tensor << endl;

  return 0;
}