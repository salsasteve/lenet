import numpy as np

def read_data(filename, dim1_size, dim2_size):
    # Assuming the data is in float32 format
    data = np.fromfile(filename, dtype=np.float32)
    if len(data) != dim1_size * dim2_size:
        raise ValueError("File size does not match the specified dimensions.")
    matrix = data.reshape(dim2_size, dim1_size)
    return matrix

def uniform_quantize(data, num_bits=8):
    """
    Applies uniform quantization to the data.

    :param data: The float32 data to be quantized.
    :param num_bits: The number of bits to use for the quantized data.
    :return: Quantized data as integers.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    data_normalized = (data - min_val) / (max_val - min_val)

    scale_factor = 2 ** num_bits - 1
    quantized_data = np.round(data_normalized * scale_factor)

    # Choose the appropriate data type based on the number of bits
    if num_bits <= 8:
        quantized_data = quantized_data.astype(np.uint8)
    elif num_bits <= 16:
        quantized_data = quantized_data.astype(np.uint16)
    else:
        quantized_data = quantized_data.astype(np.int32)

    return quantized_data

def generate_cpp_array(header_filename, matrix, variable_name="quantized_data"):
    """
    Generates a C++ array initialization string from a NumPy matrix.

    :param matrix: The NumPy matrix containing the quantized data.
    :param variable_name: The name of the C++ array variable.
    :return: A string containing the C++ code to initialize the array with the matrix data.
    """
    flat_matrix = matrix.flatten()
    array_elements = ', '.join(map(str, flat_matrix))
    cpp_file = f"const expr uint16_t {variable_name}[] = {{{array_elements}}};"
    with open(header_filename, 'w') as f:
        f.write(cpp_file)


# Example usage
input_filename = '../read_model/parameters/dense_1_bias.bin'
output_filename = '../read_model/parameters/dense_1_bias_quantized.bin'

# Read the original data
original_data = read_data(input_filename, 120, 1)

# Quantize the data
quantized_data = uniform_quantize(original_data, num_bits=16)



# Create the header file
header_filename = '../lenet_sequential/dense_1_bias_quantized.h'

generate_cpp_array(header_filename, quantized_data, variable_name="dense_1_bias_quantized")

