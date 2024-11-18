import numpy as np
import cupy as cp
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
import math


class GpuAugmentation:

    def __init__(self, n_new_images, image_path=None, image_matrix=None, height=None, width=None, channels=None):
        """
        Class for image augmentation using GPU.

        :param image_path: Path to the image file to be augmented.
        :param n_new_images: Number of augmented images to create.
        """

        # input variables
        self.n_new_images = n_new_images
        self.image_path = image_path

        # it converts image in array and it takes info
        if self.image_path is not None:
            self.cpu_image_array_1d, image_info = self.from_image_to_array()
            self.width = image_info['width']
            self.height = image_info['height']
            self.channels = image_info['channels']
            self.gpu_mean_per_channel = cp.asarray(image_info['mean_per_channel'], dtype=cp.uint8)
        else:
            self.cpu_image_array_1d = image_matrix.ravel()
            self.width = width
            self.height = height
            self.channels = channels
            self.gpu_mean_per_channel = cp.mean(image_matrix, axis=(0, 1), keepdims=True)

        # it creates one GPU version of input image and n_new_images GPU versions of output images
        self.gpu_image_input = cp.asarray(self.cpu_image_array_1d, dtype=cp.uint8)

        # for each new image it creates a list [image, image height, image width]
        self.gpu_images_outputs = [[cp.zeros(self.width * self.height * self.channels, dtype=cp.uint8),
                                    self.height, self.width, self.channels] for i in range(self.n_new_images)]

        # Parameters for CUDA functions
        self.threads_per_block = 16                    # number of threads per block

        self.block_dim = (self.threads_per_block,
                          self.threads_per_block, 1)   # block dimensions (x, y, z) for CUDA functions

        self.grid_dim = ((self.width + self.block_dim[0] - 1) // self.block_dim[0],
                         (self.height + self.block_dim[1] - 1) // self.block_dim[1],
                         1)                            # grid dimensions (x, y, z) for CUDA functions

    def from_image_to_array(self):
        """
        It converts an image (the one in self.image_path) into an array of dimensions width * height * channels.
        :return: A tuple containing:
                 - flatten_image_array: A 1D array of the image's pixel data.
                 - A dictionary with the keys:
                     - 'width': The width of the image.
                     - 'height': The height of the image.
                     - 'channels': The number of color channels in the image.
                     - 'mean_per_channel': mean of pixel values for each channel.
        """

        mode_to_channels = {"RGB": 3, "RGBA": 4, "L": 1}  # conversion from image mode (in PIL) to number of channels
        image = Image.open(self.image_path)

        width, height = image.size
        channels = mode_to_channels.get(image.mode, None)

        image_array = np.array(image)
        mean_per_channel = cp.mean(image_array, axis=(0, 1), keepdims=True)

        return image_array.ravel(),\
            {'width': width,
             'height': height,
             'channels': channels,
             'mean_per_channel': mean_per_channel[0][0]}

    def rotation(self, index, angle):
        """
        Rotate the image by a specified angle.

        :param angle: The rotation angle (must be one of 90, 180, or 270 degrees).
        :param index: The index of the image to modify in self.gpu_images_outputs.
        """

        # CUDA code for ROTATION FUNCTIONS
        rotate_image = cp.RawKernel(r'''
            extern "C"
            __global__ void rotate_image(unsigned char* input, unsigned char* output, int width, 
                                             int height, int channels, int angle) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < width && y < height) {
                    if (angle == 270) {
                        for (int c = 0; c < channels; ++c) {
                            output[x * height * channels + (height - 1 - y) * channels + c] = 
                                input[y * width * channels + x * channels + c];
                        }
                    } else {
                        if (angle == 90) {
                            for (int c = 0; c < channels; ++c) {
                                output[(width - 1 - x) * height * channels + y * channels + c] = 
                                    input[y * width * channels + x * channels + c];
                            }
                        } else {
                            for (int c = 0; c < channels; ++c) {
                                output[(height - 1 - y) * width * channels + (width - 1 - x) * channels + c] = 
                                    input[y * width * channels + x * channels + c];
                            }
                        }   
                    }
                }
            }
            ''', 'rotate_image')

        allowed_angles = [90, 180, 270]

        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))

        if angle not in allowed_angles:
            raise Exception('Not allowed angle. Possible angles are 90, 180, 270')

        if angle == 270 or angle == 90:
            # it is a transformation in which height and width exchange their values
            # (see self.gpu_images_outputs comments)
            self.gpu_images_outputs[index][1] = self.width
            self.gpu_images_outputs[index][2] = self.height

        rotate_image(self.grid_dim,
                     self.block_dim,
                     (
                        self.gpu_image_input,
                        self.gpu_images_outputs[index][0],
                        self.width,
                        self.height,
                        self.channels,
                        angle)
                     )  # grid, block and arguments

    # DONE
    def reflection(self, index, direction='vertical'):
        """
        Reflect the image along x (horizontal) or y (vertical) axes.

        :param index: The index of the image to modify in self.gpu_images_outputs.
        :param direction: Direction of the reflection ('horizontal' or 'vertical' allowed). direction='vertical'
                          by default.
        """

        # CUDA code for REFLECTION FUNCTIONS
        reflect_image = cp.RawKernel(r'''
            extern "C"
            __global__ void reflect_image(unsigned char* input, unsigned char* output, int width, 
                                             int height, int channels, int axis) {
    
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
    
                if (x < width && y < height) {
                    if (axis == 0) {
                        for (int c = 0; c < channels; ++c) {
                            output[y * width * channels + (width - 1 - x) * channels + c] = 
                                input[y * width * channels + x * channels + c];
                        }
                    } else {
                        for (int c = 0; c < channels; ++c) {
                            output[(height - 1 - y) * width * channels + x * channels + c] = 
                                input[y * width * channels + x * channels + c];
                        }
                    }
                }
            }
            ''', 'reflect_image')

        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))
        if direction != 'vertical' and direction != 'horizontal':
            raise Exception("Not allowed direction. Allowed possibilities are 'horizontal', 'vertical'")

        axis = 0 if direction == 'vertical' else 1
        reflect_image(
            self.grid_dim,
            self.block_dim,
            (
                self.gpu_image_input,
                self.gpu_images_outputs[index][0],
                self.width,
                self.height,
                self.channels,
                axis
            )
        )

    # DONE
    def brightness(self, index, percentage, add_brightness=True):
        """
        Increase (decrease) the brightness of the image.

        :param index: The index of the image to modify in self.gpu_images_outputs.
        :param percentage: Level of increase.
        :param add_brightness: True for increase, False for decrease (True by default).
        """

        # CUDA code for BRIGHTNESS FUNCTIONS
        bright_image = cp.RawKernel(r'''
            extern "C"
            __global__ void bright_image(unsigned char* input, unsigned char* output, int width, 
                                             int height, int channels, int percentage, int to_add) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                unsigned char delta, pixel_value, division;
                unsigned char percent_pixel = (unsigned char)((percentage * 255) / 100);
                

                if (x < width && y < height) {
                    if (to_add == 0) {
                        for (int c = 0; c < channels; ++c) {
                            pixel_value = input[y * width * channels + x * channels + c];
                            delta = (unsigned char)(255) - pixel_value;
                            division = (unsigned char) ((pixel_value + percent_pixel) / 255);
                            output[y * width * channels + x * channels + c] = pixel_value + 
                                                                    (1 - division) * percent_pixel + division * delta; 
                        }
                    } else {
                        for (int c = 0; c < channels; ++c) {
                            pixel_value = input[y * width * channels + x * channels + c];
                            division = (unsigned char) ((pixel_value + 255) / (percent_pixel + 255));
                            output[y * width * channels + x * channels + c] = division * (pixel_value - percent_pixel);
                        }  
                    }            
                }
            }
            ''', 'bright_image')

        allowed_percentage = [10, 20, 30, 40, 50]

        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))
        if percentage not in allowed_percentage:
            raise Exception('Not allowed percentage. Allowed values are: 10, 20, 30, 40, 50.')

        to_add = 0 if add_brightness else 1

        bright_image(
            self.grid_dim,
            self.block_dim,
            (
                self.gpu_image_input,
                self.gpu_images_outputs[index][0],
                self.width,
                self.height,
                self.channels,
                percentage,
                to_add
            )
        )

    # DONE
    def contrast(self, index, contrast_factor):
        """
        Increase (decrease) the brightness of the image.

        :param index: The index of the image to modify in self.gpu_images_outputs.
        :param contrast_factor: Number in range [0, 5] with only one digit allowed.
        """

        contrast = 1
        # CUDA code for CONTRAST FUNCTIONS
        contrast_image = cp.RawKernel(r'''
            extern "C"
            __global__ void contrast_image(unsigned char* input, unsigned char* output, int width, 
                                             int height, int channels, unsigned char* means, int brightness) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                int check;

                float pixel_value;
                float contrast_factor = (float)brightness / 10;

                if (x < width && y < height) {
                    for (int c = 0; c < channels; ++c) {
                        pixel_value = (float) input[y * width * channels + x * channels + c];
                        pixel_value = (pixel_value - (float) means[c]) * contrast_factor + (float) means[c];
                        check = (int) ((pixel_value + 510) / 765);

                        output[y * width * channels + x * channels + c] = (unsigned char) (pixel_value * (1 - check) 
                                                                            + 255 * check); 
                    }
                }
            }
            ''', 'contrast_image')

        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))

        contrast = contrast_factor * 10
        if contrast in range(-1, 51):
            contrast_image(
                self.grid_dim,
                self.block_dim,
                (
                    self.gpu_image_input,
                    self.gpu_images_outputs[index][0],
                    self.width,
                    self.height,
                    self.channels,
                    self.gpu_mean_per_channel,
                    int(contrast)
                )
            )
        else:
            raise Exception('Contrast out of range. It should be in range [0, 5] (only one digit allowed).')

    def noise(self, index, noise_type='gaussian', mean=0, sigma=20, noise_fraction=0.3):

        allowed_fraction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        allowed_types = ['gaussian', 'salt_pepper']
        input_image_len = len(self.cpu_image_array_1d)

        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))
        if noise_type not in allowed_types:
            raise Exception("Not allowed value for noise_type. Allowed types are: 'gaussian', 'salt_pepper'.")
        if noise_fraction not in allowed_fraction:
            raise Exception('Not allowed value for noise_fraction. Allowed values are: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9')

        if noise_type == 'gaussian':
            gaussian_noise = cp.random.normal(mean, sigma, input_image_len).astype('uint8')
            self.gpu_images_outputs[index][0] = cp.add(self.gpu_image_input, gaussian_noise)

        if noise_type == 'salt_pepper':
            num_elem_to_modify = int(input_image_len * noise_fraction / 2)
            pepper_indices = cp.random.choice(input_image_len, num_elem_to_modify, replace=False)
            salt_indices = cp.random.choice(input_image_len, num_elem_to_modify, replace=False)

            self.gpu_images_outputs[index][0] = self.gpu_image_input
            self.gpu_images_outputs[index][0][pepper_indices] = 0
            self.gpu_images_outputs[index][0][salt_indices] = 255

    def shear(self, index, x_shear_factor, y_shear_factor):

        n_new_x_pixel = int(self.width * x_shear_factor / 100)
        delta_x = n_new_x_pixel / self.height

        n_new_y_pixel = int(self.height * y_shear_factor / 100)
        delta_y = n_new_y_pixel / self.width

        # CUDA code for SHEAR FUNCTIONS
        shear_image = cp.RawKernel(r'''
            extern "C"
            __global__ void shear_image(unsigned char* input, unsigned char* output, int width, 
                                             int height, int channels, float delta_x, int new_width, float delta_y, int new_height) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                int new_x, new_y;
                
                if (x < width && y < height) {
                    new_x = x + (int) (delta_x * y);
                    new_y = y + (int) (delta_y * x);
                
                    for (int c = 0; c < channels; ++c) {
                        output[new_y * new_width * channels + new_x * channels + c] = input[y * width * channels + x * channels + c];
                    }
                }
            }
            ''', 'shear_image')

        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))

        self.gpu_images_outputs[index][0] = cp.zeros((self.width + n_new_x_pixel) * (self.height + n_new_y_pixel) * self.channels, dtype=cp.uint8)
        self.gpu_images_outputs[index][1] += n_new_y_pixel
        self.gpu_images_outputs[index][2] += n_new_x_pixel

        shear_image(
            self.grid_dim,
            self.block_dim,
            (
                self.gpu_image_input,
                self.gpu_images_outputs[index][0],
                self.width,
                self.height,
                self.channels,
                cp.float32(delta_x),
                self.width + n_new_x_pixel,
                cp.float32(delta_y),
                self.height + n_new_y_pixel
            )
        )

    # DONE
    def occlusion(self, index, width_percent, height_percent):

        occlude_image = cp.RawKernel(r'''
            extern "C"
            __global__ void occlude_image(unsigned char* input, unsigned char* output, int width, 
                                        int height, int channels, int x_lower, int x_upper, int y_lower, int y_upper) {
    
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                
                int x_b, y_b;
    
                if (x < width && y < height) {
                    x_b = (int) ((x_lower + width) / (x + width)) + (int) ((x + width) / (x_upper + width));
                    y_b = (int) ((y_lower + height) / (y + height)) + (int) ((y + height) / (y_upper + height));
                    
                    for (int c = 0; c < channels; ++c) {
                        output[y * width * channels + x * channels + c] = input[y * width * channels + x * channels + c] 
                            * (int) ((x_b + y_b + 1) / 2);
                    }            
                }
            }
            ''', 'occlude_image')

        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))

        x_size = int(width_percent * self.width / 100)
        y_size = int(height_percent * self.height / 100)

        left_y_vertex = int(np.random.uniform(0, self.height - y_size))
        left_x_vertex = int(np.random.uniform(0, self.width - x_size))

        right_x_vertex = left_x_vertex + x_size
        right_y_vertex = left_y_vertex + y_size

        occlude_image(
            self.grid_dim,
            self.block_dim,
            (
                self.gpu_image_input,
                self.gpu_images_outputs[index][0],
                self.width,
                self.height,
                self.channels,
                left_x_vertex,
                right_x_vertex,
                left_y_vertex,
                right_y_vertex
            )
        )

    # DONE
    def gray_scale(self, index):
        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))
        if self.channels != 3:
            raise Exception('Only RGB images (3 channels) can be transform. Number of channels: ', self.channels)

        self.gpu_images_outputs[index][0] = cp.zeros(self.width * self.height, dtype=cp.uint8)
        self.gpu_images_outputs[index][3] = 1

        gray_scale = cp.RawKernel(r'''
            extern "C"
            __global__ void gray_scale(unsigned char* input, unsigned char* output, int width,
                                            int height, int input_channels) {
    
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                unsigned char gray_level;
                
    
                if (x < width && y < height) {    
                
                    gray_level = (unsigned char) (0.299 * input[y * width * input_channels + x * input_channels]  + 
                                 0.587 * input[y * width * input_channels + x * input_channels + 1] + 
                                 0.114 * input[y * width * input_channels + x * input_channels + 2]);
                                  
                    output[y * width + x] = gray_level;          
                }
            }
            ''', 'gray_scale')

        gray_scale(
            self.grid_dim,
            self.block_dim,
            (
                self.gpu_image_input,
                self.gpu_images_outputs[index][0],
                self.width,
                self.height,
                self.channels
            )
        )

    # DONE
    def channel_shuffle(self, index):
        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))

        shuffle = cp.RawKernel(r'''
            extern "C"
            __global__ void shuffle(unsigned char* input, unsigned char* output, int width,
                                            int height, int channels) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                unsigned char first, second, third;

                if (x < width && y < height) {
                    first = input[y * width * channels + x * channels];
                    second = input[y * width * channels + x * channels + 1];
                    third = input[y * width * channels + x * channels + 2]; 
                    
                    output[y * width * channels + x * channels] = second;
                    output[y * width * channels + x * channels + 1] = third; 
                    output[y * width * channels + x * channels + 2] = first;      
                }
            }
            ''', 'shuffle')

        shuffle(
            self.grid_dim,
            self.block_dim,
            (
                self.gpu_image_input,
                self.gpu_images_outputs[index][0],
                self.width,
                self.height,
                self.channels
            )
        )

    # DONE
    def crop(self, index, vertical_percent, horizontal_percent):

        # CUDA code for CROP FUNCTIONS
        crop_image = cp.RawKernel(r'''
            extern "C"
            __global__ void crop_image(unsigned char* input, unsigned char* output, int width, 
                                        int height, int channels, int x_vertex, int y_vertex,int crop_width,
                                        int crop_height) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < crop_width && y < crop_height) {
                    int input_x = x + x_vertex;
                    int input_y = y + y_vertex;
                    
                    for (int c = 0; c < channels; ++c) {
                        output[y * crop_width * channels + x * channels + c] = 
                                                            input[input_y * width * channels + input_x * channels + c];
                    }            
                }
            }
            ''', 'crop_image')

        allowed_percentage = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        if index > len(self.gpu_images_outputs):
            raise Exception('Index out of range. Maximum allowed for index: ', len(self.gpu_images_outputs))
        if vertical_percent not in allowed_percentage:
            raise Exception('Not allowed percentage. Allowed values are: 10, 20, 30, 40, 50, 60, 70, 80, 90.')
        if horizontal_percent not in allowed_percentage:
            raise Exception('Not allowed percentage. Allowed values are: 10, 20, 30, 40, 50, 60, 70, 80, 90.')

        new_width = int(self.width * horizontal_percent / 100)
        new_height = int(self.height * vertical_percent / 100)

        upper_y_vertex = int(np.random.uniform(0, self.height - new_height))
        upper_x_vertex = int(np.random.uniform(0, self.width - new_width))

        gpu_temp_image = cp.zeros(new_width * new_height * self.channels, dtype=cp.uint8)

        crop_image(
            self.grid_dim,
            self.block_dim,
            (
                self.gpu_image_input,
                gpu_temp_image,
                self.width,
                self.height,
                self.channels,
                upper_x_vertex,
                upper_y_vertex,
                new_width,
                new_height
            )
        )

        self.__private_resize(gpu_temp_image, new_width, new_height, index)

    # DONE
    def __private_resize(self, gpu_cropped_image, cropped_width, cropped_height, index):

        # CUDA code for RESIZE FUNCTIONS
        resize_image = cp.RawKernel(r'''
            extern "C"
            __global__ void resize_image(unsigned char* input, unsigned char* output, int width, 
                              int height, int channels, int crop_width, int crop_height) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
            
                if (x < width && y < height) {
                    float x_scale = (float)crop_width / width; 
                    float y_scale = (float)crop_height / height;
            
                    float origin_x = x * x_scale;
                    float origin_y = y * y_scale;
            
                    int x1 = (int)floor(origin_x);
                    int x2 = min(x1 + 1, crop_width - 1);
                    //int x2 = (int)ceil(origin_x);
                    
                    int y1 = (int)floor(origin_y);
                    int y2 = min(y1 + 1, crop_height - 1);
                    //int y2 = (int)ceil(origin_y);
            
                    float w1 = (x2 - origin_x) * (y2 - origin_y);
                    float w2 = (origin_x - x1) * (y2 - origin_y);
                    float w3 = (x2 - origin_x) * (origin_y - y1);
                    float w4 = (origin_x - x1) * (origin_y - y1);
            
                    for (int c = 0; c < channels; ++c) {
                        float pixel_value = 
                            (w1 * input[y1 * crop_width * channels + x1 * channels + c] +
                             w2 * input[y1 * crop_width * channels + x2 * channels + c] +
                             w3 * input[y2 * crop_width * channels + x1 * channels + c] +
                             w4 * input[y2 * crop_width * channels + x2 * channels + c]);
            
                        output[y * width * channels + x * channels + c] = (unsigned char)fminf(fmaxf(pixel_value, 0.0f), 255.0f);
                        //output[y * width * channels + x * channels + c] = (unsigned char)pixel_value;
                    }
                }
            }
            ''', 'resize_image')

        resize_image(
            self.grid_dim,
            self.block_dim,
            (
                gpu_cropped_image,
                self.gpu_images_outputs[index][0],
                self.width,
                self.height,
                self.channels,
                cropped_width,
                cropped_height
            )
        )

    def from_array_to_image(self):
        """
        Transform every array in self.gpu_images_outputs into a PIL image.

        :return: A list of images.
        """

        images = []
        for image_array in self.gpu_images_outputs:
            image = image_array[0].get()
            if image_array[3] == 3:
                image = image.reshape((image_array[1], image_array[2], image_array[3]))
            else:
                image = image.reshape((image_array[1], image_array[2]))
            images.append(Image.fromarray(image))

        return images


class MyAugmentation:
    def __init__(self, images, width, height, channels, batch_size):
        self.images = images
        self.width = width
        self.height = height
        self.channels = channels
        self.output_channels = channels
        self.batch_size = batch_size
        self.input_batches_on_gpu, self.output_batches_on_gpu, self.n_batches = self.load_batches_on_gpu()

        self.augmented_images = []

        # Parameters for CUDA functions
        self.threads_per_block = 16  # number of threads per block

        self.block_dim = (self.threads_per_block,
                          self.threads_per_block, 1)  # block dimensions (x, y, z) for CUDA functions

        self.grid_dim = ((self.width + self.block_dim[0] - 1) // self.block_dim[0],
                         (self.height + self.block_dim[1] - 1) // self.block_dim[1],
                         1)  # grid dimensions (x, y, z) for CUDA functions

    def load_batches_on_gpu(self):
        i = 0
        n_images = len(self.images)
        n_elem_in_batch = 0
        flatten_images = []
        flatten_output_images = []
        batches_on_gpu = []
        output_batches_on_gpu = []
        means_per_channel = []
        all_zeroes = np.zeros(self.width * self.height * self.channels, dtype=cp.uint8)

        for image in self.images:
            n_elem_in_batch += 1
            i += 1

            means_per_channel.append(np.mean(image, axis=(0, 1), keepdims=True))
            flatten_images.append(image.ravel())
            flatten_output_images.append(all_zeroes)

            if i % self.batch_size == 0 or i % n_images == 0:

                flatten_images = np.concatenate(flatten_images)
                flatten_output_images = np.concatenate(flatten_output_images)

                batch = cp.asarray(flatten_images, dtype=cp.uint8)
                zero_batch = cp.asarray(flatten_output_images, dtype=cp.uint8)
                means = cp.asarray(np.mean(means_per_channel, axis=(0, 1), keepdims=True))

                batches_on_gpu.append({'batch': batch, 'length': n_elem_in_batch, 'means': means[0][0][0]})
                output_batches_on_gpu.append({'batch': zero_batch, 'length': n_elem_in_batch})
                flatten_images = []
                flatten_output_images = []
                means_per_channel = []
                n_elem_in_batch = 0

        return batches_on_gpu, output_batches_on_gpu, len(batches_on_gpu)

    def flip(self, direction='vertical'):
        # CUDA code for REFLECTION FUNCTIONS
        reflect_image = cp.RawKernel(r'''
            extern "C"
            __global__ void reflect_image(unsigned char* input, unsigned char* output, int width, 
                                             int height, int channels, int batch_size, int axis) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < width && y < height) {
                    for (int i = 0; i < batch_size; i++) {
                        if (axis == 0) {
                            for (int c = 0; c < channels; ++c) {
                                output[i * width * height * channels + y * width * channels + (width - 1 - x) * channels + c] = 
                                    input[i * width * height * channels + y * width * channels + x * channels + c];
                            }
                        } else {
                            for (int c = 0; c < channels; ++c) {
                                output[i * width * height * channels + (height - 1 - y) * width * channels + x * channels + c] = 
                                    input[i * width * height * channels + y * width * channels + x * channels + c];
                            }
                        }
                    }
                }
            }
            ''', 'reflect_image')

        axis = 0 if direction == 'vertical' else 1

        for i in range(self.n_batches):
            reflect_image(
                self.grid_dim,
                self.block_dim,
                (
                    self.input_batches_on_gpu[i]['batch'],
                    self.output_batches_on_gpu[i]['batch'],
                    self.width,
                    self.height,
                    self.channels,
                    self.input_batches_on_gpu[i]['length'],
                    axis
                )
            )

    def to_gray(self):
        gray_scale = cp.RawKernel(r'''
            extern "C"
            __global__ void gray_scale(unsigned char* input, unsigned char* output, int width,
                                            int height, int input_channels, int batch_size) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                int batch_dim = width * height * input_channels;
                int b_pos;
                unsigned char gray_level;


                if (x < width && y < height) {    
                    for (int i = 0; i < batch_size; i++){
                        b_pos = i * batch_dim;
                        gray_level = (unsigned char) (
                                    0.299 * input[b_pos + y * width * input_channels + x * input_channels]  + 
                                    0.587 * input[b_pos + y * width * input_channels + x * input_channels + 1] + 
                                    0.114 * input[b_pos + y * width * input_channels + x * input_channels + 2]
                                );

                        output[b_pos + y * width + x] = gray_level; 
                    }         
                }
            }
            ''', 'gray_scale')

        for i in range(self.n_batches):
            gray_scale(
                self.grid_dim,
                self.block_dim,
                (
                    self.input_batches_on_gpu[i]['batch'],
                    self.output_batches_on_gpu[i]['batch'],
                    self.width,
                    self.height,
                    self.channels,
                    self.input_batches_on_gpu[i]['length']
                )
            )

        self.output_channels = 1

    def channel_shuffle(self):
        shuffle = cp.RawKernel(r'''
            extern "C"
            __global__ void shuffle(unsigned char* input, unsigned char* output, int width,
                                            int height, int channels, int batch_size) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                unsigned char first, second, third;
                
                if (x < width && y < height) {
                    for (int i = 0; i< batch_size; i++) {
                        first = input[i * width * height * channels + y * width * channels + x * channels];
                        second = input[i * width * height * channels + y * width * channels + x * channels + 1];
                        third = input[i * width * height * channels + y * width * channels + x * channels + 2]; 
    
                        output[i * width * height * channels + y * width * channels + x * channels] = second;
                        output[i * width * height * channels + y * width * channels + x * channels + 1] = third; 
                        output[i * width * height * channels + y * width * channels + x * channels + 2] = first;    
                    }      
                }
            }
            ''', 'shuffle')

        for i in range(self.n_batches):
            shuffle(
                self.grid_dim,
                self.block_dim,
                (
                    self.input_batches_on_gpu[i]['batch'],
                    self.output_batches_on_gpu[i]['batch'],
                    self.width,
                    self.height,
                    self.channels,
                    self.input_batches_on_gpu[i]['length'],
                )
            )

    def brightness(self, percentage, to_add=True):
        # CUDA code for BRIGHTNESS FUNCTIONS
        bright_image = cp.RawKernel(r'''
            extern "C"
            __global__ void bright_image(unsigned char* input, unsigned char* output, int width, 
                                             int height, int channels, int batch_size, int percentage, int to_add) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                unsigned char delta, pixel_value, division;
                unsigned char percent_pixel = (unsigned char)((percentage * 255) / 100);


                if (x < width && y < height) {
                    for (int i = 0; i< batch_size; i++) {
                        if (to_add == 0) {
                            for (int c = 0; c < channels; ++c) {
                                pixel_value = input[i * width * height * channels + y * width * channels + x * channels + c];
                                delta = (unsigned char)(255) - pixel_value;
                                division = (unsigned char) ((pixel_value + percent_pixel) / 255);
                                output[i * width * height * channels + y * width * channels + x * channels + c] = pixel_value + 
                                                                        (1 - division) * percent_pixel + division * delta; 
                            }
                        } else {
                            for (int c = 0; c < channels; ++c) {
                                pixel_value = input[i * width * height * channels + y * width * channels + x * channels + c];
                                division = (unsigned char) ((pixel_value + 255) / (percent_pixel + 255));
                                output[i * width * height * channels + y * width * channels + x * channels + c] = division * (pixel_value - percent_pixel);
                            }  
                        }   
                    }         
                }
            }
            ''', 'bright_image')

        for i in range(self.n_batches):
            bright_image(
                self.grid_dim,
                self.block_dim,
                (
                    self.input_batches_on_gpu[i]['batch'],
                    self.output_batches_on_gpu[i]['batch'],
                    self.width,
                    self.height,
                    self.channels,
                    self.input_batches_on_gpu[i]['length'],
                    percentage,
                    0 if to_add else 1
                )
            )

    def contrast(self, contrast_factor):
        # CUDA code for CONTRAST FUNCTIONS
        contrast_image = cp.RawKernel(r'''
            extern "C"
            __global__ void contrast_image(unsigned char* input, unsigned char* output, int width, 
                                             int height, int channels, int batch_size, unsigned char* means, int brightness) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                int check;

                float pixel_value;
                float contrast_factor = (float)brightness / 10;

                if (x < width && y < height) {
                    for (int i = 0; i < batch_size; i++) {
                        for (int c = 0; c < channels; ++c) {
                            pixel_value = (float) input[i * width * height * channels + y * width * channels + x * channels + c];
                            pixel_value = (pixel_value - (float) means[c]) * contrast_factor + (float) means[c];
                            check = (int) ((pixel_value + 510) / 765);
    
                            output[i * width * height * channels + y * width * channels + x * channels + c] = (unsigned char) (pixel_value * (1 - check) 
                                                                                + 255 * check); 
                        }
                    }
                }
            }
            ''', 'contrast_image')

        for i in range(self.n_batches):
            contrast_image(
                self.grid_dim,
                self.block_dim,
                (
                    self.input_batches_on_gpu[i]['batch'],
                    self.output_batches_on_gpu[i]['batch'],
                    self.width,
                    self.height,
                    self.channels,
                    self.input_batches_on_gpu[i]['length'],
                    self.input_batches_on_gpu[i]['means'],
                    contrast_factor
                )
            )

    def occlusion(self, width_percent, height_percent):
        occlude_image = cp.RawKernel(r'''
            extern "C"
            __global__ void occlude_image(unsigned char* input, unsigned char* output, int width, 
                                        int height, int channels, int batch_size, int x_lower, int x_upper, int y_lower, int y_upper) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                int x_b, y_b;

                if (x < width && y < height) {
                    for (int i = 0; i < batch_size; i++) {
                        x_b = (int) ((x_lower + width) / (x + width)) + (int) ((x + width) / (x_upper + width));
                        y_b = (int) ((y_lower + height) / (y + height)) + (int) ((y + height) / (y_upper + height));
    
                        for (int c = 0; c < channels; ++c) {
                            output[i * width * height * channels + y * width * channels + x * channels + c] = 
                            input[i * width * height * channels + y * width * channels + x * channels + c] * (int) ((x_b + y_b + 1) / 2);
                        }  
                    }   
                }
            }
            ''', 'occlude_image')

        x_size = int(width_percent * self.width / 100)
        y_size = int(height_percent * self.height / 100)

        left_y_vertex = int(np.random.uniform(0, self.height - y_size))
        left_x_vertex = int(np.random.uniform(0, self.width - x_size))

        right_x_vertex = left_x_vertex + x_size
        right_y_vertex = left_y_vertex + y_size

        for i in range(self.n_batches):
            occlude_image(
                self.grid_dim,
                self.block_dim,
                (
                    self.input_batches_on_gpu[i]['batch'],
                    self.output_batches_on_gpu[i]['batch'],
                    self.width,
                    self.height,
                    self.channels,
                    self.input_batches_on_gpu[i]['length'],
                    left_x_vertex,
                    right_x_vertex,
                    left_y_vertex,
                    right_y_vertex
                )
            )

    def crop(self, vertical_percent, horizontal_percent):

        # CUDA code for CROP FUNCTIONS
        crop_image = cp.RawKernel(r'''
            extern "C"
            __global__ void crop_image(unsigned char* input, unsigned char* output, int width, 
                                        int height, int channels, int batch_size, int x_vertex, int y_vertex,int crop_width,
                                        int crop_height) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < crop_width && y < crop_height) {
                    int input_x = x + x_vertex;
                    int input_y = y + y_vertex;
                    
                    for (int i = 0; i < batch_size; i++) {
                        for (int c = 0; c < channels; ++c) {
                            output[i * width * height * channels + y * width * channels + x * channels + c] = 
                            input[i * width * height * channels + input_y * width * channels + input_x * channels + c];
                        }    
                    } 
                }
            }
            ''', 'crop_image')

        new_width = int(self.width * horizontal_percent / 100)
        new_height = int(self.height * vertical_percent / 100)

        upper_y_vertex = int(np.random.uniform(0, self.height - new_height))
        upper_x_vertex = int(np.random.uniform(0, self.width - new_width))

        for i in range(self.n_batches):
            crop_image(
                self.grid_dim,
                self.block_dim,
                (
                    self.input_batches_on_gpu[i]['batch'],
                    self.output_batches_on_gpu[i]['batch'],
                    self.width,
                    self.height,
                    self.channels,
                    self.input_batches_on_gpu[i]['length'],
                    upper_x_vertex,
                    upper_y_vertex,
                    new_width,
                    new_height
                )
            )

    def resize(self, cropped_width, cropped_height):
        # CUDA code for RESIZE FUNCTIONS
        resize_image = cp.RawKernel(r'''
            extern "C"
            __global__ void resize_image(unsigned char* input, unsigned char* output, int width, 
                              int height, int channels, int batch_size,  int crop_width, int crop_height) {

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x < width && y < height) {
                    float x_scale = (float)crop_width / width; 
                    float y_scale = (float)crop_height / height;

                    float origin_x = x * x_scale;
                    float origin_y = y * y_scale;

                    int x1 = (int)floor(origin_x);
                    int x2 = min(x1 + 1, crop_width - 1);
                    //int x2 = (int)ceil(origin_x);

                    int y1 = (int)floor(origin_y);
                    int y2 = min(y1 + 1, crop_height - 1);
                    //int y2 = (int)ceil(origin_y);

                    float w1 = (x2 - origin_x) * (y2 - origin_y);
                    float w2 = (origin_x - x1) * (y2 - origin_y);
                    float w3 = (x2 - origin_x) * (origin_y - y1);
                    float w4 = (origin_x - x1) * (origin_y - y1);
                    
                    for (int i = 0; i < batch_size; i++) {
                        for (int c = 0; c < channels; ++c) {
                            float pixel_value = 
                                (w1 * input[i * width * height * channels + y1 * width * channels + x1 * channels + c] +
                                 w2 * input[i * width * height * channels + y1 * width * channels + x2 * channels + c] +
                                 w3 * input[i * width * height * channels + y2 * width * channels + x1 * channels + c] +
                                 w4 * input[i * width * height * channels + y2 * width * channels + x2 * channels + c]);
    
                            output[i * width * height * channels + y * width * channels + x * channels + c] = (unsigned char)fminf(fmaxf(pixel_value, 0.0f), 255.0f);
                            //output[y * width * channels + x * channels + c] = (unsigned char)pixel_value;
                        }
                    }
                }
            }
            ''', 'resize_image')

        for i in range(self.n_batches):
            resize_image(
                self.grid_dim,
                self.block_dim,
                (
                    self.input_batches_on_gpu[i]['batch'],
                    self.output_batches_on_gpu[i]['batch'],
                    self.width,
                    self.height,
                    self.channels,
                    self.input_batches_on_gpu[i]['length'],
                    cropped_width,
                    cropped_height
                )
            )

    def crop_and_resize(self, vertical_percent, horizontal_percent):
        new_width = int(self.width * horizontal_percent / 100)
        new_height = int(self.height * vertical_percent / 100)

        self.crop(vertical_percent, horizontal_percent)
        self.resize(new_width, new_height)

    def gaussian_noise(self, mean=0, std=20):
        length = self.height * self.width * self.channels
        for i in range(self.n_batches):
            batch_length = length * self.input_batches_on_gpu[i]['length']
            gaussian_noise = cp.random.normal(mean, std, batch_length).astype('float32')
            noisy_batch = cp.add(self.input_batches_on_gpu[i]['batch'].astype('float32'), gaussian_noise)
            self.output_batches_on_gpu[i]['batch'] = cp.clip(noisy_batch, 0, 255).astype('uint8')

    def from_batch_to_image(self):
        all_images = []
        image_dim = self.output_channels * self.width * self.height

        for i in range(self.n_batches):
            images = self.output_batches_on_gpu[i]['batch']
            length = self.output_batches_on_gpu[i]['length']

            for j in range(length):
                start = j * self.width * self.height * self.channels
                stop = start + self.width * self.height * self.output_channels
                image = images[start:stop]
                if self.output_channels == 1:
                    image = image.reshape((self.height, self.width))
                else:
                    image = image.reshape((self.height, self.width, self.output_channels))

                all_images.append(image.get())

        # all_images = cp.concatenate(all_images)
        self.augmented_images = all_images

    def show_image(self, index):
        image = Image.fromarray(self.augmented_images[index])
        image.show()


class ParallelAlbumentation:
    def __init__(self, data, batch_size, max_workers=None):
        self.data = data
        self.batch_size = batch_size
        self.n_batches = math.ceil(len(data) / batch_size)
        self.max_workers = max_workers if max_workers is not None else os.cpu_count()
        self.batches = self.return_batches()
        self.results = [None] * self.n_batches

    def return_batches(self):
        batches = []
        temp_batch = []

        n_elem_in_batch = 0
        i = 0
        n_images = len(self.data)

        for image in self.data:
            n_elem_in_batch += 1
            i += 1
            temp_batch.append(image)

            if i % self.batch_size == 0 or i % n_images == 0:
                batches.append(temp_batch)
                temp_batch = []
                n_elem_in_batch = 0

        return batches

    @staticmethod
    def process_batch(transformation, batch_data):
        return [transformation(image=image)['image'] for image in batch_data]


    def transform_applier(self, transformation):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_batch, transformation, batch): batch
                for batch in self.batches
            }

            results = []
            for future in future_to_batch:
                results.append(future.result())
        return results

    def horizontal_flip(self):
        return self.transform_applier(A.HorizontalFlip(p=1))

    def vertical_flip(self):
        return self.transform_applier(A.VerticalFlip(p=1))

    def to_gray(self):
        return self.transform_applier(A.ToGray(p=1))

    def channel_shuffle(self):
        return self.transform_applier(A.ChannelShuffle(p=1))

    def brightness(self, b_lim, to_add=True):
        b_lim = b_lim if to_add else -1 * b_lim
        return self.transform_applier(A.RandomBrightnessContrast(brightness_limit=(b_lim, b_lim),
                                                                 contrast_limit=(0, 0),
                                                                 p=1))

    def contrast(self, c_min, c_max):
        return self.transform_applier(A.ColorJitter(contrast=(c_min, c_max), p=1))

    def occlusion(self, height, width):
        return self.transform_applier(A.CoarseDropout(hole_height_range=(height, height),
                                                      hole_width_range=(width, width),
                                                      p=1))

    def crop(self, dim):
        return self.transform_applier(A.RandomSizedCrop((dim, dim), size=(375, 500)))

    def gaussian_noise(self, var_min, var_max):
        return self.transform_applier(A.GaussNoise(var_limit=(var_min, var_max), p=1))

    @staticmethod
    def show_image(images, index):
        flatten_images = []
        for batch_images in images:
            for image in batch_images:
                flatten_images.append(image)
        image = Image.fromarray(flatten_images[index])
        image.show()

