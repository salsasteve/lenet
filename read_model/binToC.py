import struct

def binary_file_to_c_float_array(file_path, rows, cols, array_name="matrix"):
    # Read the binary file
    with open(file_path, "rb") as file:
        binary_data = file.read()

    # Calculate the total number of floats in the matrix
    num_floats = rows * cols
    if len(binary_data) != num_floats * 4:  # 4 bytes per float
        raise ValueError("File size does not match the specified matrix dimensions")

    # Convert binary data to floats
    floats = struct.unpack(f"{num_floats}f", binary_data)

    # Start generating the C array declaration
    c_code = f"float {array_name}[{rows}][{cols}] = {{\n"

    # Convert each float to a C-compatible format and arrange them in matrix format
    for i, value in enumerate(floats):
        if i % cols == 0:
            c_code += "    { "  # New row
        c_code += f"{value:.6f}, "
        if (i + 1) % cols == 0:
            c_code = c_code.rstrip(', ') + " },\n"  # End of row

    c_code += "};\n"  # End of matrix

    return c_code

# Example usage
file_path = "parameters/conv2d_1_bias.bin"  # Replace with the path to your binary file
rows, cols = 4, 4  # Replace with the actual dimensions of your matrix
c_code = binary_file_to_c_float_array(file_path, rows, cols)
print(c_code)
