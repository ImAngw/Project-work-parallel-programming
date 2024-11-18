# Image Augmentation with CUDA

This project implements image augmentation functions using CUDA to accelerate operations. The code is divided into three main files:

- `main.py`: Contains the main code that collects and measures the execution times of the augmentation functions.
- `augmented_functions.py`: Defines the classes and methods used in `main.py` to perform the augmentation operations.
- `cudaFunctions.cu`: Contains the CUDA code for the augmentation functions implemented in `augmented_functions.py`.
## Dependencies

This project uses the following libraries:
- **Albumentations**: A Python library for fast and optimized image augmentation transformations.
- **CuPy**: A library that enables GPU-accelerated operations in Python, used to speed up augmentation functions with CUDA.

## Dataset
To use the program, you need to provide a dataset consisting of images all with the same dimensions. The images should be converted into a matrix format before processing.
- All images should have the same size (e.g., 500x375 pixels).
- The dataset should be in matrix form, where each image is represented as a matrix of pixel values.

# How to use
Run main.py to start the augmentation process and measure execution times.
