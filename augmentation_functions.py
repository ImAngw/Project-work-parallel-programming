import numpy as np
import cupy as cp
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import albumentations as A
import math


class CustomAugmentation:
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

