# load 2 feature maps sets from 2 files and compare them
# features maps are 28x28
# there are 6 feature maps
import numpy as np

# Load the feature maps
cpp_file_path = "./feature_maps.bin"
cpp_feature_maps = np.fromfile(cpp_file_path, dtype=np.float32).reshape(-1, 28, 28)

# Load the feature maps
python_file_path = "./convolution_output.bin"
python_feature_maps = np.fromfile(python_file_path, dtype=np.float32).reshape(-1, 28, 28)

# # Compare the feature maps
# for i in range(len(cpp_feature_maps)):
#     print("Feature map", i)
#     print("CPP")
#     print(cpp_feature_maps[i])
#     print("Python")
#     print(python_feature_maps[i])
#     print("Difference")
#     print(cpp_feature_maps[i] - python_feature_maps[i])
#     print()

print("Feature map", 0)
print("CPP")
print(cpp_feature_maps[0])
print("Python")
print(python_feature_maps[0])
print("Difference")
print(cpp_feature_maps[0] - python_feature_maps[0])
print()

# print first 10 elements of the first feature map
print("CPP")
print(cpp_feature_maps[0][:10])
print("Python")
print(python_feature_maps[0][:10])
print("Difference")
print(cpp_feature_maps[0][:10] - python_feature_maps[0][:10])   
print()

# print dimensions of the feature maps
print("CPP")
print(cpp_feature_maps.shape)
print("Python")
print(python_feature_maps.shape)

# round all values to 4 decimal places
cpp_feature_maps = np.round(cpp_feature_maps, 5)
python_feature_maps = np.round(python_feature_maps, 5)

print("Total Difference")
print(np.sum(cpp_feature_maps - python_feature_maps))
print("Average Difference")
print(np.mean(cpp_feature_maps - python_feature_maps))

# iterate till you find a difference and wait for the user to press enter

for i in range(len(cpp_feature_maps)):
    print("Feature map", i)
    print("CPP")
    print(cpp_feature_maps[i])
    print("Python")
    print(python_feature_maps[i])
    print("Difference")
    print(cpp_feature_maps[i] - python_feature_maps[i])
    print()
    if np.sum(cpp_feature_maps[i] - python_feature_maps[i]) != 0:
        input("Press Enter to continue...")
