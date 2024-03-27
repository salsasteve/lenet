#ifndef CONFIG_H
#define CONFIG_H
#include <string>

namespace config {


// Convolutional layer 1
std::string conv2d_1_weights = "../read_model/parameters/conv2d_1_weights.bin";
std::string conv2d_1_bias = "../read_model/parameters/conv2d_1_bias.bin";

// Convolutional layer 2
std::string conv2d_2_weights = "../read_model/parameters/conv2d_2_weights.bin";
std::string conv2d_2_bias = "../read_model/parameters/conv2d_2_bias.bin";

// Dense layer 1
std::string dense_1_weights = "../read_model/parameters/dense_1_weights.bin";
std::string dense_1_bias = "../read_model/parameters/dense_1_bias.bin";

// Dense layer 2
std::string dense_2_weights = "../read_model/parameters/dense_2_weights.bin";
std::string dense_2_bias = "../read_model/parameters/dense_2_bias.bin";

// Dense layer 3
std::string dense_3_weights = "../read_model/parameters/dense_3_weights.bin";
std::string dense_3_bias = "../read_model/parameters/dense_3_bias.bin";

// output csv


#endif // CONFIG_H
