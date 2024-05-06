#include <iostream>
#include <vector>
#include "MNISTLoader.h"
#include "config.h"
#include "read_model_weights.h"
#include "pooling.h"
#include "dense_layer.h"
#include "convolution_cuda.h"
#include "pooling_cuda.h"
#include <fstream>

using namespace std;
using Image = vector<vector<float>>;
using Batch = vector<Image>;
// 2D vector to hold the kernels for each layer
// [height][width]
using Kernel = vector<vector<float>>;
// 3D vector to hold the kernels for each layer
// [number of kernels][height][width
using Kernels = vector<Kernel>;
// 4D vector to hold the kernels for each layer
// [number of kernels][depth][height][width]
// depth is the number of input feature maps
// number of kernels is the number of output feature maps
using DeepKernels = vector<Kernels>;
// 2D vector to hold the feature maps for each layer
// [height][width]
using FeatureMap = vector<vector<float>>;
// 3D vector to hold the feature maps for each layer
// [channels][height][width]
// channels is the number of output feature maps
using FeatureMaps = vector<FeatureMap>;

vector<float> flatten(const FeatureMaps &featureMaps)
{
    vector<float> flattened;
    int dim1=featureMaps.size();
    int dim2=featureMaps[0].size();
    int dim3=featureMaps[0][0].size();
    for (int j=0; j<dim2; j++)
    {
        for (int k=0; k<dim3; k++)
        {
            for (int i=0; i<dim1; i++)
            {
                flattened.push_back(featureMaps[i][j][k]);
            }
        }
    }
    return flattened;
}

std::vector<float> softmax1D(const std::vector<float>& v) {
    float sum = 0.0f;
    std::vector<float> output(v.size());

    // Compute the sum of exp(values) for the vector
    for (auto val : v) {
        sum += std::exp(val);
    }

    // Compute softmax for each value in the vector
    for (size_t i = 0; i < v.size(); ++i) {
        output[i] = std::exp(v[i]) / sum;
    }
    return output;
}

// Hyperbolic tangent function applied to a 1D vector
std::vector<float> tanh1D(const std::vector<float>& v) {
    std::vector<float> output;
    for (auto val : v) {
        output.push_back(std::tanh(val));
    }
    return output;
}

int main()
{
    // LeNet-5 Input Layer configuration
    InputLayerConfig inputLayer = {28, 28, 10, 28 * 28};
    // LeNet-5 Layer 1 configuration
    ConvLayerConfig layer1Config = {5, 5, 1, 6, 1, 1, 2};
    string conv2d_1_bias = "../read_model/parameters/conv2d_1_bias.bin";
    string conv2d_1_weights = "../read_model/parameters/conv2d_1_weights.bin";
    string conv2d_2_weights = "../read_model/parameters/conv2d_2_weights.bin";
    string dense_1_weights = "../read_model/parameters/dense_1_weights.bin";
    string dense_1_bias = "../read_model/parameters/dense_1_bias.bin";
    string dense_2_weights = "../read_model/parameters/dense_2_weights.bin";
    string dense_2_bias = "../read_model/parameters/dense_2_bias.bin";
    string dense_3_weights = "../read_model/parameters/dense_3_weights.bin";
    string dense_3_bias = "../read_model/parameters/dense_3_bias.bin";
    string conv2d_2_bias = "../read_model/parameters/conv2d_2_bias.bin";

    // create 3d vector to hold the images with aliases
    Image image(inputLayer.inputHeight, vector<float>(inputLayer.inputWidth, 0.0));
    vector<Image> allImages(inputLayer.numOfImageForInference, image);

    allImages = getMNISTImages(inputLayer.numOfImageForInference);
    vector<float> output = {};
   

    for(int i = 0; i < inputLayer.numOfImageForInference; i++)
    {
        
	    vector<Image> input(1, allImages[i]);
        // Create 6 kernels for the first layer of LeNet-5 4D vector with aliases  
        DeepKernels layer1DeepKernels = LoadConv2DWeights(conv2d_1_weights,
                                          layer1Config.kernelHeight,
                                          layer1Config.kernelWidth,
					                      layer1Config.kernelDepth,
                                          layer1Config.kernelsCount);
        vector<float> biases = LoadBias(conv2d_1_bias, layer1Config.kernelsCount);

        FeatureMaps layer1FeatureMaps = convolve2dDeep_GPU(input, layer1DeepKernels, biases, layer1Config.horizontalStride, layer1Config.paddingAmount);
        FeatureMaps layer1PooledFeatureMaps = averagePooling3D_CUDA(layer1FeatureMaps, 2, 2);
        
        ConvLayerConfig layer2Config = {5, 5, 6, 16, 1, 1, 0};

        // Create 16 kernels for the second layer of LeNet-5 4D vector with aliases
        Kernel layer2Kernel(layer2Config.kernelHeight, vector<float>(layer2Config.kernelWidth, 0.0));
        Kernels layer2Kernels(layer2Config.kernelDepth, layer2Kernel);
        DeepKernels layer2DeepKernels(layer2Config.kernelsCount, layer2Kernels);

        
        layer2DeepKernels = LoadConv2DWeights(conv2d_2_weights,
                                            layer2Config.kernelHeight,
                                            layer2Config.kernelWidth,
					                        layer2Config.kernelDepth,
                                            layer2Config.kernelsCount);

        vector<float> biases2(layer1Config.kernelsCount);
        biases2 = LoadBias(conv2d_2_bias, layer2Config.kernelsCount);

        FeatureMaps layer2FeatureMaps = convolve2dDeep_GPU(layer1PooledFeatureMaps, layer2DeepKernels, biases2, layer2Config.horizontalStride, layer2Config.paddingAmount);
        // Check the dimensions of the feature maps
        std::cout << "Layer 2 feature maps dimensions: " << layer2FeatureMaps.size() << "x" << layer2FeatureMaps[0].size() << "x" << layer2FeatureMaps[0][0].size() << endl;
        FeatureMaps layer2PooledFeatureMaps = averagePooling3D_CUDA(layer2FeatureMaps, 2, 2);

        // Check the dimensions of the feature maps
        std::cout << "Layer 2 feature maps dimensions: " << layer2PooledFeatureMaps.size() << "x" << layer2PooledFeatureMaps[0].size() << "x" << layer2PooledFeatureMaps[0][0].size() << endl;

        // Flatten the feature maps
        vector<float> flattenedFeatures = flatten(layer2PooledFeatureMaps);
	    std::cout << "Flattened features count: " << flattenedFeatures.size() << endl;

        // Load the weights for the first dense layer
        // 400 input neurons, 120 output neurons
        // 400x120 weights
        
        TwoD dense1Weights = LoadDenseWeights(dense_1_weights, 120, 400);

        // Load the biases for the first dense layer
        // 120 biases
        vector<float> dense1Biases = LoadBias(dense_1_bias, 120);

        // Perform the matrix multiplication
        vector<float> dense1Layer = dense(flattenedFeatures, dense1Biases, dense1Weights, 120);
        // Apply the activation function
        vector<float> dense1Activated = tanh1D(dense1Layer);

        // print dimensions of the dense layer
        std::cout << "Dense layer dimensions: " << dense1Activated.size() << endl;

        // Load the weights for the second dense layer
        // 120 input neurons, 84 output neurons
        // 120x84 weights
        TwoD dense2Weights = LoadDenseWeights(dense_2_weights, 84, 120);

        // Load the biases for the second dense layer
        // 84 biases
        vector<float> dense2Biases = LoadBias(dense_2_bias, 84);

        // Perform the matrix multiplication
        vector<float> dense2Layer = dense(dense1Activated, dense2Biases, dense2Weights, 84);

        // Apply the activation function
        vector<float> dense2Activated = tanh1D(dense2Layer);

        // print dimensions of the dense layer
        std::cout << "Dense layer dimensions: " << dense2Activated.size() << endl;

        // Load the weights for the third dense layer
        // 84 input neurons, 10 output neurons
        // 84x10 weights
        TwoD dense3Weights = LoadDenseWeights(dense_3_weights, 10, 84);

        // Load the biases for the third dense layer
        // 10 biases
        
        vector<float> dense3Biases = LoadBias(dense_3_bias, 10);

        // Perform the matrix multiplication
        vector<float> dense3Layer = dense(dense2Activated, dense3Biases, dense3Weights, 10);

        // Apply the activation function
        vector<float> dense3Activated = softmax1D(dense3Layer);

        // print dimensions of the dense layer
        std::cout << "Dense layer dimensions: " << dense3Activated.size() << endl;

        // Print the output
        // Convert one hot encoding to a single digit
        int maxIndex = 0;
        float maxVal = 0.0;
        for (int i = 0; i < dense3Activated.size(); ++i)
        {
            if (dense3Activated[i] > maxVal)
            {
                maxVal = dense3Activated[i];
                maxIndex = i;
            }
        }
        //save output to array
        output.push_back(maxIndex);
        
        
    }
    //print output
    for(int i = 0; i < inputLayer.numOfImageForInference; i++)
    {
        std::cout << "Output: " << output[i] << endl;
    }
    std::cout << "Number of images: " << inputLayer.numOfImageForInference << endl;

    return 0;
}
